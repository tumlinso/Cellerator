#include <Cellerator/biology/coordinates.hh>
#include <Cellerator/biology/feature_binding.hh>
#include <Cellerator/nn/attention.hh>
#include <Cellerator/nn/features.hh>
#include <Cellerator/nn/programs.hh>
#include <Cellerator/nn/readouts.hh>
#include <Cellerator/nn/supervision.hh>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <torch/autograd.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nn = cellerator::nn;
namespace bio = cellerator::biology;
using Tensor = at::Tensor;

namespace {

constexpr double fp16_atol = 4e-3;
constexpr double fp16_rtol = 4e-3;

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

bool finite(const Tensor& value) {
    return at::isfinite(value).all().item<bool>();
}

void close(const Tensor& actual, const Tensor& expected,
           double atol, double rtol, const std::string& label) {
    check(actual.sizes() == expected.sizes(), label + ": shape mismatch");
    check(actual.scalar_type() == expected.scalar_type(), label + ": dtype mismatch");
    check(actual.device() == expected.device(), label + ": device mismatch");
    if (!at::allclose(actual, expected, rtol, atol, false)) {
        const auto error = (actual.to(at::kFloat) - expected.to(at::kFloat)).abs();
        const float maximum = error.numel() == 0 ? 0.0f : error.max().item<float>();
        throw std::runtime_error(label + ": max_abs_error=" + std::to_string(maximum));
    }
}

nn::AttentionOptions attention_options(std::int64_t width,
                                      std::int64_t heads,
                                      nn::AttentionKernel kernel) {
    nn::AttentionOptions options;
    options.width = width;
    options.heads = heads;
    options.dropout = 0.0;
    options.kernel = kernel;
    return options;
}

void append_gradient_inputs(std::vector<Tensor>& inputs,
                            const nn::LinearParameters& layer) {
    inputs.push_back(layer.weight);
    if (layer.bias.defined()) inputs.push_back(layer.bias);
}

std::vector<Tensor> attention_gradient_inputs(
    const Tensor& input, const nn::AttentionParameters& parameters) {
    std::vector<Tensor> inputs{input};
    append_gradient_inputs(inputs, parameters.query);
    append_gradient_inputs(inputs, parameters.key);
    append_gradient_inputs(inputs, parameters.value);
    append_gradient_inputs(inputs, parameters.output);
    return inputs;
}

void prepared_attention_matches_aten(const at::Device& device) {
    constexpr std::int64_t width = 4;
    constexpr std::int64_t heads = 2;
    nn::Parameters owner(device, at::kHalf);
    auto group = owner.group();
    nn::AttentionParameters parameters(group, width);
    nn::RelationAttention aten_attention(
        parameters, attention_options(width, heads, nn::AttentionKernel::aten));
    nn::RelationAttention csr_attention(
        parameters, attention_options(width, heads,
                                      nn::AttentionKernel::cellerator_csr));
    check(aten_attention.parameters().query.weight.is_same(
              csr_attention.parameters().query.weight),
          "ATen and CSR attention must share the same parameter tensors");

    const auto order = bio::OccurrenceCoordinates<std::int64_t>::from_ids(
        {0, 1, 2, 3, 4, 5}, {0, 3, 6}, {}, 19);
    const auto relation = nn::PreparedRelation::from_csr(
        nn::CsrSupport{
            {0, 3, 5, 5, 7, 9, 9},
            {0, 0, 1, 0, 2, 3, 4, 3, 5},
        },
        order.order_token(), order.order_token(), 19, device);
    relation.validate_for(order.order_token(), order.order_token(), 6, 6, 19, device);

    const auto represented = at::randn(
        {6, width}, at::TensorOptions().dtype(at::kHalf).device(device));
    auto aten_input = represented.clone();
    aten_input.requires_grad_(true);
    auto csr_input = represented.clone();
    csr_input.requires_grad_(true);

    const auto aten_output = aten_attention(aten_input, relation);
    const auto csr_output = csr_attention(csr_input, relation);
    check(finite(aten_output) && finite(csr_output),
          "prepared attention outputs must be finite");
    close(csr_output, aten_output, fp16_atol, fp16_rtol,
          "prepared FP16 CSR/ATen forward");

    const auto aten_loss = aten_output.to(at::kFloat).square().mean();
    const auto csr_loss = csr_output.to(at::kFloat).square().mean();
    const auto gradient_inputs = attention_gradient_inputs(
        aten_input, aten_attention.parameters());
    const auto aten_gradients = torch::autograd::grad({aten_loss}, gradient_inputs);
    auto csr_gradient_inputs = attention_gradient_inputs(
        csr_input, csr_attention.parameters());
    const auto csr_gradients = torch::autograd::grad({csr_loss}, csr_gradient_inputs);
    check(aten_gradients.size() == csr_gradients.size(),
          "prepared attention gradient counts must match");
    for (std::size_t index = 0; index < aten_gradients.size(); ++index) {
        check(finite(aten_gradients[index]) && finite(csr_gradients[index]),
              "prepared attention gradients must be finite");
        close(csr_gradients[index], aten_gradients[index], fp16_atol, fp16_rtol,
              "prepared FP16 CSR/ATen VJP " + std::to_string(index));
    }
}

std::pair<Tensor, Tensor> build_supervised_frozen_composition(
    const at::Device& device,
    nn::Parameters& owner,
    Tensor& input_values) {
    constexpr std::int64_t batch = 2;
    constexpr std::int64_t features_per_group = 3;
    constexpr std::int64_t width = 8;

    auto group = owner.group();
    nn::FeatureEncoderOptions encoder_options;
    encoder_options.vocabulary_size = 8;
    encoder_options.width = width;
    encoder_options.value_size = 1;
    nn::FeatureEncoder encoder(group, encoder_options);
    nn::RelationAttention attention(
        group, attention_options(width, 2, nn::AttentionKernel::aten));
    nn::InteractionPrograms programs(group, width, 3, 2);
    nn::GatedInteractor interactor(group, width, 12, width, true);

    bio::FeatureBindingOptions binding_options;
    binding_options.domain = "gene";
    binding_options.model_capacity = 8;
    auto binding = bio::FeatureBinding<std::string>::from_ids(
        {"g0", "g1", "g2"}, {0, 2, 4}, binding_options);
    nn::FeatureReadout<std::string> readout(group, binding, width, 1, 2);
    const std::vector<std::string> requested_features{"g2", "g0"};
    const auto targets = nn::prepare_targets(binding, requested_features, device);

    const auto feature_order = bio::OccurrenceCoordinates<std::string>::from_ids(
        {"cell-a/g0", "cell-a/g1", "cell-a/g2",
         "cell-b/g0", "cell-b/g1", "cell-b/g2"},
        {0, 3, 6}, {}, 23);
    const auto program_order = bio::OccurrenceCoordinates<std::string>::from_ids(
        {"module-a", "module-b", "empty",
         "module-a", "module-b", "empty"},
        {0, 3, 6}, {}, 23);
    const auto relation = nn::PreparedRelation::from_csr(
        nn::CsrSupport{
            {0, 3, 5, 5, 7, 9, 9},
            {0, 1, 1, 1, 2, 3, 4, 4, 5},
        },
        feature_order.order_token(), program_order.order_token(), 23, device);

    nn::ObservationCoordinates<std::string> observations{
        {"cell-a", "cell-b"}, {0, 1, 2}};
    const auto target_values = at::randn(
        {batch, features_per_group},
        at::TensorOptions().dtype(at::kHalf).device(device));

    // All components are constructed in one explicit owner and then frozen.
    // Autograd still follows the supplied feature values through the full model.
    group.set_trainable(false);

    nn::FeatureInput input;
    input.ids = at::tensor(std::vector<std::int64_t>{0, 2, 4, 4, 0, 2},
                           at::TensorOptions().dtype(at::kLong))
                    .reshape({batch, features_per_group}).to(device);
    input.values = input_values;
    input.observed = at::ones(
        {batch, features_per_group},
        at::TensorOptions().dtype(at::kBool).device(device));

    const auto encoded = encoder(input);
    const auto attended = attention(encoded);
    nn::ProgramCall program_call;
    program_call.relation = &relation;
    program_call.feature_order = feature_order.order_token();
    program_call.program_order = program_order.order_token();
    program_call.structural_generation = 23;
    const auto program_result = programs(
        attended.reshape({batch * features_per_group, width}), program_call);
    const auto program_features = program_result.program_features.reshape({batch, 3, width});
    const auto feature_updates = program_result.feature_updates.reshape(
        {batch, features_per_group, width});
    const auto pooled_programs = interactor(program_features);
    const auto context = attended + feature_updates + pooled_programs.unsqueeze(1);

    const auto predictions = readout.query(context, targets, observations);
    using Coordinates = nn::ObservationCoordinates<std::string>;
    using Payload = nn::QueryPayload<std::string, Coordinates>;
    const nn::Prediction<Tensor, Payload> prediction{
        predictions,
        Payload{std::span<const std::string>(targets.canonical_ids()),
                std::cref(observations), {}}};
    const auto aligned = nn::align_prediction_targets(
        prediction, target_values, targets.alignment(), observations);
    const auto losses = (aligned.first.to(at::kFloat)
                         - aligned.second.to(at::kFloat)).square();
    const auto reduced = nn::reduce_prediction_loss(
        losses, nn::SupervisionUnit::observation, observations);
    return {reduced.first, reduced.second};
}

void frozen_full_composition_preserves_input_gradient(const at::Device& device) {
    constexpr std::int64_t batch = 2;
    constexpr std::int64_t features_per_group = 3;
    nn::Parameters owner(device, at::kHalf);
    auto input_values = at::randn(
        {batch, features_per_group},
        at::TensorOptions().dtype(at::kHalf).device(device));
    input_values.requires_grad_(true);
    const auto reduced = build_supervised_frozen_composition(
        device, owner, input_values);
    check(reduced.first.dim() == 0 && finite(reduced.first),
          "supervised frozen composition must return a finite scalar loss");
    check(reduced.second.item<std::int64_t>() == 4,
          "supervised query must align the requested feature and observation rows");
    reduced.first.backward();
    check(input_values.grad().defined() && finite(input_values.grad()),
          "frozen model parameters must preserve finite input gradients");
    check(input_values.grad().abs().sum().item<float>() > 0.0f,
          "frozen model must not detach its supplied feature values");
    for (const auto& parameter : owner.torch_parameters()) {
        check(!parameter.requires_grad(), "the shared model parameter group must be frozen");
        check(!parameter.grad().defined(), "frozen model parameters must not accumulate gradients");
    }
}

void overlapping_programs_produce_finite_vjps(const at::Device& device) {
    constexpr std::int64_t width = 8;
    nn::Parameters owner(device, at::kHalf);
    nn::InteractionPrograms programs(owner.group(), width, 3, 2);
    auto features = at::randn(
        {4, width}, at::TensorOptions().dtype(at::kHalf).device(device));
    features.requires_grad_(true);

    const auto feature_order = bio::OccurrenceCoordinates<std::string>::from_ids(
        {"g0", "g1", "g2", "g3"}, {}, {}, 31);
    const auto program_order = bio::OccurrenceCoordinates<std::string>::from_ids(
        {"module-a", "module-b", "empty"}, {}, {}, 31);
    const auto relation = nn::PreparedRelation::from_csr(
        nn::CsrSupport{{0, 3, 5, 5}, {0, 1, 1, 1, 2}},
        feature_order.order_token(), program_order.order_token(), 31, device);

    nn::ProgramCall call;
    call.relation = &relation;
    call.feature_order = feature_order.order_token();
    call.program_order = program_order.order_token();
    call.structural_generation = 31;
    const auto output = programs(features, call);
    check(output.program_features.sizes() == at::IntArrayRef({3, width}),
          "overlapping program feature shape mismatch");
    check(output.feature_updates.sizes() == features.sizes(),
          "overlapping program update shape mismatch");
    check(finite(output.program_features) && finite(output.feature_updates),
          "overlapping program outputs must be finite");
    const auto loss = output.program_features.to(at::kFloat).square().sum()
                    + output.feature_updates.to(at::kFloat).square().sum();
    loss.backward();
    check(features.grad().defined() && finite(features.grad()),
          "overlapping programs must produce finite input VJPs");
    check(programs.program_queries().grad().defined()
              && finite(programs.program_queries().grad()),
          "overlapping programs must produce finite shared-query VJPs");
    for (const auto& parameter : owner.torch_parameters()) {
        check(parameter.grad().defined() && finite(parameter.grad()),
              "overlapping program attention parameters must receive finite VJPs");
    }
}

void run_device_tests() {
    const at::Device device(at::kCUDA, 0);
    c10::cuda::CUDAGuard guard(device);
    cudaDeviceProp properties{};
    const auto property_status = cudaGetDeviceProperties(&properties, device.index());
    check(property_status == cudaSuccess,
          std::string("could not query selected GPU properties: ")
              + cudaGetErrorString(property_status));
    std::ostringstream uuid_text;
    uuid_text << std::hex << std::setfill('0');
    for (std::size_t index = 0; index < sizeof(properties.uuid.bytes); ++index) {
        if (index == 4 || index == 6 || index == 8 || index == 10) {
            uuid_text << '-';
        }
        uuid_text << std::setw(2)
                  << static_cast<unsigned int>(
                         static_cast<unsigned char>(properties.uuid.bytes[index]));
    }
    prepared_attention_matches_aten(device);
    frozen_full_composition_preserves_input_gradient(device);
    overlapping_programs_produce_finite_vjps(device);
    std::cout << "native CUDA model composition passed: uuid=" << uuid_text.str()
              << " name=" << properties.name << " sm=" << properties.major
              << '.' << properties.minor << " device=" << device.str()
              << "; FP16 CSR parity/VJPs, frozen-input gradients, query supervision, "
                 "and overlapping-program VJPs\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2 || std::string(argv[1]) != "--device") {
        std::cout << "CUDA model qualification is opt-in. Run with --device inside an admitted "
                     "Device Scheduler job.\n";
        return 0;
    }
    try {
        run_device_tests();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native CUDA model qualification failed: " << error.what() << '\n';
        return 1;
    }
}

