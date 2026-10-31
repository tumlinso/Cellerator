#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/nn/programs.hh>

#include <ATen/ATen.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace nn = cellerator::nn;
namespace bio = cellerator::biology;

namespace {

at::Tensor random_tensor(std::initializer_list<std::int64_t> shape,
                         bool requires_grad = false) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
    auto result = at::randn(shape, options);
    return requires_grad ? result.set_requires_grad(true) : result;
}

bool all_finite(const at::Tensor& value) {
    return at::isfinite(value).all().item<bool>();
}

void assert_all_parameters_receive_finite_gradients(const nn::Parameters& owner) {
    for (const auto& parameter : owner.torch_parameters()) {
        assert(parameter.grad().defined());
        assert(all_finite(parameter.grad()));
    }
}

void dense_programs_preserve_gradients() {
    at::manual_seed(301);
    nn::Parameters owner(at::Device(at::kCPU), at::kFloat);
    nn::InteractionPrograms model(owner.group(), 4, 3, 2);
    auto features = random_tensor({2, 5, 4}, true);

    const auto output = model(features);

    assert(output.program_features.sizes() == at::IntArrayRef({2, 3, 4}));
    assert(output.feature_updates.sizes() == at::IntArrayRef({2, 5, 4}));
    assert(features.requires_grad());
    assert(all_finite(output.program_features));
    assert(all_finite(output.feature_updates));
    (output.program_features.square().mean()
     + output.feature_updates.square().mean()).backward();
    assert(features.grad().defined());
    assert(all_finite(features.grad()));
    assert(model.program_queries().grad().defined());
    assert(all_finite(model.program_queries().grad()));
    assert_all_parameters_receive_finite_gradients(owner);

    const std::array<std::int64_t, 2> repeated_bank{1, 1};
    const std::array<std::int64_t, 2> repeated_offsets{0, 2};
    nn::ProgramCall repeated_call;
    repeated_call.program_bank_indices = std::span<const std::int64_t>(repeated_bank);
    repeated_call.program_offsets = std::span<const std::int64_t>(repeated_offsets);
    const auto repeated = model(features.detach().select(0, 0), repeated_call);
    assert(repeated.program_features.sizes() == at::IntArrayRef({2, 4}));
    assert(at::allclose(repeated.program_features.select(0, 0),
                        repeated.program_features.select(0, 1)));
}

void prepared_overlapping_programs_keep_duplicates_and_empty_rows() {
    at::manual_seed(302);
    const auto features = bio::OccurrenceCoordinates<std::string>::from_ids(
        {"g0", "g1", "g2", "g3", "g4"}, {0, 3, 5, 5}, {}, 7);
    const auto programs = bio::OccurrenceCoordinates<std::string>::from_ids(
        {"p0", "p1", "empty", "p0", "p1", "empty",
         "p0", "p1", "empty"},
        {0, 3, 6, 9}, {}, 7);
    const auto relation = nn::PreparedRelation::from_csr(
        nn::CsrSupport{
            {0, 3, 5, 5, 6, 6, 6, 6, 6, 6},
            {0, 0, 1, 1, 2, 3},
        },
        features.order_token(), programs.order_token(), 7);
    assert(relation.edge_count() == 6);
    assert(relation.transpose().edge_count() == relation.edge_count());

    nn::Parameters owner(at::Device(at::kCPU), at::kFloat);
    nn::InteractionPrograms model(owner.group(), 4, 3, 2);
    auto values = random_tensor({5, 4}, true);
    nn::ProgramCall call;
    call.relation = &relation;
    call.feature_order = features.order_token();
    call.program_order = programs.order_token();
    call.structural_generation = 7;

    const auto output = model(values, call);

    assert(output.program_features.sizes() == at::IntArrayRef({9, 4}));
    assert(output.feature_updates.sizes() == at::IntArrayRef({5, 4}));
    assert(at::allclose(output.program_features.select(0, 2), at::zeros({4})));
    assert(at::allclose(output.program_features.select(0, 5), at::zeros({4})));
    assert(at::allclose(output.program_features.select(0, 6), at::zeros({4})));
    assert(at::allclose(output.program_features.select(0, 7), at::zeros({4})));
    assert(at::allclose(output.program_features.select(0, 8), at::zeros({4})));
    assert(at::allclose(output.feature_updates.select(0, 4), at::zeros({4})));

    const auto local_features = bio::OccurrenceCoordinates<std::string>::from_ids(
        {"g0", "g1", "g2"}, {}, {}, 7);
    const auto local_programs = bio::OccurrenceCoordinates<std::string>::from_ids(
        {"p0", "p1", "empty"}, {}, {}, 7);
    const auto local_relation = nn::PreparedRelation::from_csr(
        nn::CsrSupport{{0, 3, 5, 5}, {0, 0, 1, 1, 2}},
        local_features.order_token(), local_programs.order_token(), 7);
    nn::ProgramCall local_call;
    local_call.relation = &local_relation;
    local_call.feature_order = local_features.order_token();
    local_call.program_order = local_programs.order_token();
    local_call.structural_generation = 7;
    const auto local_output = model(values.detach().narrow(0, 0, 3), local_call);
    assert(at::allclose(output.program_features.narrow(0, 0, 3),
                        local_output.program_features));
    assert(at::allclose(output.feature_updates.narrow(0, 0, 3),
                        local_output.feature_updates));

    (output.program_features.square().sum()
     + output.feature_updates.square().sum()).backward();
    assert(values.grad().defined() && all_finite(values.grad()));
    assert(model.program_queries().grad().defined());
    assert(all_finite(model.program_queries().grad()));
    assert_all_parameters_receive_finite_gradients(owner);

    const auto deduplicated_relation = nn::PreparedRelation::from_csr(
        nn::CsrSupport{
            {0, 2, 4, 4, 5, 5, 5, 5, 5, 5},
            {0, 1, 1, 2, 3},
        },
        features.order_token(), programs.order_token(), 7);
    auto deduplicated_call = call;
    deduplicated_call.relation = &deduplicated_relation;
    const auto deduplicated = model(values.detach(), deduplicated_call);
    assert(!at::allclose(output.program_features.select(0, 0),
                         deduplicated.program_features.select(0, 0)));

    const auto stale = bio::OccurrenceCoordinates<std::string>::from_ids(
        {"g0", "g1", "g2", "g3", "g4"}, {0, 3, 5, 5}, {}, 7);
    call.feature_order = stale.order_token();
    bool rejected_stale_order = false;
    try {
        static_cast<void>(model(values.detach(), call));
    } catch (const std::invalid_argument&) {
        rejected_stale_order = true;
    }
    assert(rejected_stale_order);
}

void gated_interactor_keeps_legacy_path_and_conditions_gate_logits() {
    at::manual_seed(303);
    nn::Parameters owner(at::Device(at::kCPU), at::kFloat);
    auto group = owner.group();
    nn::GatedInteractor model(group, 3, 5, 2);
    auto participants = random_tensor({2, 4, 3}, true);

    const auto hidden = at::tanh(model.encoder()(participants));
    const auto weights = at::sigmoid(model.gate()(participants));
    auto pooled = (weights * hidden).sum(-2);
    auto expected = model.joint_output()(
        at::silu(model.joint_hidden()(at::silu(pooled))));
    const auto actual = model(participants);
    assert(at::allclose(actual, expected, 0.0, 0.0));

    bool rejected_unconfigured_context = false;
    try {
        static_cast<void>(model(participants, random_tensor({2, 2})));
    } catch (const std::invalid_argument&) {
        rejected_unconfigured_context = true;
    }
    assert(rejected_unconfigured_context);

    nn::Linear gate_conditioner(group, 2, 1, false);
    auto context = random_tensor({2, 2}, true);
    auto condition = [&gate_conditioner](const at::Tensor& input) {
        return gate_conditioner(input).unsqueeze(-2);
    };
    const auto conditioned = model(participants, context, condition);
    conditioned.square().sum().backward();
    assert(participants.grad().defined() && all_finite(participants.grad()));
    assert(context.grad().defined() && all_finite(context.grad()));
    assert(gate_conditioner.weight().grad().defined());
    assert(all_finite(gate_conditioner.weight().grad()));

    auto wrong_broadcast = [](const at::Tensor&) {
        return at::zeros({1, 2, 4, 1}, at::TensorOptions().dtype(at::kFloat));
    };
    bool rejected_wrong_broadcast = false;
    try {
        static_cast<void>(model(participants.detach(), context.detach(),
                                wrong_broadcast));
    } catch (const std::invalid_argument&) {
        rejected_wrong_broadcast = true;
    }
    assert(rejected_wrong_broadcast);

    nn::GatedInteractor projected(owner.group(), 3, 5, 2, false, 2);
    auto projected_context = random_tensor({2, 2}, true);
    const auto projected_output = projected(participants.detach(), projected_context);
    projected_output.square().sum().backward();
    assert(projected_context.grad().defined() && all_finite(projected_context.grad()));
}

} // namespace

int main() {
    dense_programs_preserve_gradients();
    prepared_overlapping_programs_keep_duplicates_and_empty_rows();
    gated_interactor_keeps_legacy_path_and_conditions_gate_logits();
}

