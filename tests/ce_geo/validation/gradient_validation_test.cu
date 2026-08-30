#include "../../../src/compute/architecture/providers/nvidia/sm70/edge_value_gradient.cu"
#include "../../../src/compute/architecture/providers/nvidia/sm70/exchange_program.cc"
#include "../../../src/compute/architecture/providers/nvidia/sm70/segment_backward_integration.cu"
#include "../../../src/compute/architecture/providers/nvidia/sm70/transpose_relation_apply.cu"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace sm70 =
    cellerator::compute::architecture::providers::nvidia::sm70;
namespace projection = cellerator::compute::projection;
namespace segment = cellerator::compute::segment;
namespace execution = cellerator::execution;

namespace {

[[noreturn]] void fail(const std::string &message) {
    std::cerr << "gradient validation failure: " << message << '\n';
    std::exit(EXIT_FAILURE);
}

void require(bool condition, const std::string &message) {
    if (!condition) fail(message);
}

void require_cuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess)
        fail(std::string(operation) + ": " + cudaGetErrorString(status));
}

void require_close(double actual, double expected, double tolerance,
    const std::string &label) {
    if (!std::isfinite(actual) || !std::isfinite(expected)
        || std::abs(actual - expected) > tolerance)
        fail(label + " actual=" + std::to_string(actual)
            + " expected=" + std::to_string(expected));
}

template<typename T>
class device_buffer {
  public:
    explicit device_buffer(std::size_t count) : count_(count) {
        require_cuda(cudaMalloc(reinterpret_cast<void **>(&data_),
                         count_ * sizeof(T)),
            "cudaMalloc");
    }

    ~device_buffer() {
        if (data_ != nullptr) cudaFree(data_);
    }

    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;

    T *data() const noexcept { return data_; }

    void upload(const std::vector<T> &values) {
        require(values.size() == count_, "device upload size mismatch");
        require_cuda(cudaMemcpy(data_, values.data(), count_ * sizeof(T),
                         cudaMemcpyHostToDevice),
            "cudaMemcpy host to device");
    }

    std::vector<T> download() const {
        std::vector<T> values(count_);
        require_cuda(cudaMemcpy(values.data(), data_, count_ * sizeof(T),
                         cudaMemcpyDeviceToHost),
            "cudaMemcpy device to host");
        return values;
    }

  private:
    T *data_ = nullptr;
    std::size_t count_ = 0u;
};

std::vector<__half> to_half(const std::vector<float> &values) {
    std::vector<__half> result(values.size());
    std::transform(values.begin(), values.end(), result.begin(),
        [](float value) { return __float2half(value); });
    return result;
}

double half_value(__half value) {
    return static_cast<double>(__half2float(value));
}

execution::axis_identity make_axis(std::uint32_t seed) {
    execution::axis_identity axis{};
    axis.domain = {seed, 1u};
    axis.order = {seed + 10u, 2u};
    axis.geometry = {seed + 20u, 3u};
    axis.partition = {seed + 30u, 4u};
    return axis;
}

execution::dense_tensor_view make_matrix(void *data,
    const execution::device_location &location,
    const execution::axis_identity &row_axis,
    const execution::axis_identity &column_axis, std::uint64_t rows,
    std::uint32_t columns) {
    execution::dense_tensor_view view{};
    view.data = data;
    view.location = location;
    view.value_type = execution::numeric_type::f32;
    view.rank = 2u;
    view.axes[0] = row_axis;
    view.axes[1] = column_axis;
    view.shape[0] = rows;
    view.shape[1] = columns;
    view.stride[0] = static_cast<execution::i64>(columns);
    view.stride[1] = 1;
    return view;
}

double relation_objective(const std::vector<sm70::logical_relation_edge_v1> &edges,
    const std::vector<double> &edge_values, const std::vector<double> &source,
    const std::vector<double> &destination_gradient,
    std::uint32_t dense_width) {
    double objective = 0.0;
    for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
        const auto &edge = edges[edge_index];
        for (std::uint32_t column = 0u; column < dense_width; ++column)
            objective += edge_values[edge_index]
                * source[static_cast<std::size_t>(edge.source_index)
                    * dense_width + column]
                * destination_gradient[
                    static_cast<std::size_t>(edge.destination_index)
                        * dense_width
                    + column];
    }
    return objective;
}

void validate_transpose_and_edge_values(cudaStream_t stream) {
    constexpr std::uint32_t source_count = 4u;
    constexpr std::uint32_t destination_count = 3u;
    constexpr std::uint32_t width = 5u;
    const std::vector<sm70::logical_relation_edge_v1> edges{
        {101u, 0u, 1u}, {205u, 2u, 0u}, {307u, 0u, 2u},
        {409u, 3u, 1u}, {511u, 1u, 2u}};
    std::vector<sm70::target_edge_placement_v1> forward(edges.size());
    std::vector<sm70::target_edge_placement_v1> transpose(edges.size());
    const sm70::transpose_cover_request_v1 cover_request{edges.data(),
        edges.size(), source_count, destination_count,
        projection::logical_edge_id_width_v1::u32, forward.data(),
        forward.size(), transpose.data(), transpose.size()};
    require(sm70::build_transpose_cover_v1(cover_request)
            == sm70::transpose_cover_status_v1::success,
        "transpose cover preparation failed");

    const std::vector<__half> edge_values_half =
        to_half({0.5f, -1.25f, 0.75f, 1.5f, -0.375f});
    const std::vector<__half> destination_gradient_half = to_half({
        0.25f, -0.5f, 1.0f, 0.75f, -1.25f,
        -0.75f, 0.125f, 0.5f, 1.25f, 0.375f,
        1.5f, -0.25f, -0.625f, 0.875f, 0.5f});
    device_buffer<sm70::target_edge_placement_v1> d_transpose(transpose.size());
    d_transpose.upload(transpose);
    device_buffer<__half> d_edge_values(edge_values_half.size());
    d_edge_values.upload(edge_values_half);
    device_buffer<__half> d_destination_gradient(
        destination_gradient_half.size());
    d_destination_gradient.upload(destination_gradient_half);
    device_buffer<float> d_source_gradient(source_count * width);
    const sm70::transpose_relation_apply_request_v1 transpose_request{
        d_transpose.data(), d_edge_values.data(), edges.size(),
        d_destination_gradient.data(), destination_count, source_count, width,
        d_source_gradient.data(), stream};
    require(sm70::enqueue_transpose_relation_apply_v1(transpose_request)
            == sm70::transpose_relation_apply_status_v1::success,
        "transpose relation launch failed");
    require_cuda(cudaStreamSynchronize(stream), "transpose synchronize");
    const auto source_gradient = d_source_gradient.download();

    std::vector<double> edge_values(edges.size());
    std::transform(edge_values_half.begin(), edge_values_half.end(),
        edge_values.begin(), half_value);
    std::vector<double> destination_gradient(destination_gradient_half.size());
    std::transform(destination_gradient_half.begin(),
        destination_gradient_half.end(), destination_gradient.begin(),
        half_value);
    const std::vector<double> source{
        0.25, -0.75, 1.0, 0.5, -0.125,
        1.25, 0.375, -0.5, 0.75, 0.25,
        -1.0, 0.625, 0.875, -0.25, 1.5,
        0.5, 1.0, -1.25, 0.125, -0.375};
    constexpr double epsilon = 1.0e-6;
    for (std::size_t index = 0; index < source.size(); ++index) {
        auto plus = source;
        auto minus = source;
        plus[index] += epsilon;
        minus[index] -= epsilon;
        const double finite_difference =
            (relation_objective(edges, edge_values, plus,
                 destination_gradient, width)
                - relation_objective(edges, edge_values, minus,
                    destination_gradient, width))
            / (2.0 * epsilon);
        require_close(source_gradient[index], finite_difference, 2.0e-5,
            "transpose source gradient " + std::to_string(index));
    }

    std::vector<sm70::support_logical_edge_v1> support_edges(edges.size());
    for (std::size_t index = 0; index < edges.size(); ++index) {
        support_edges[index].logical_edge_id = {
            edges[index].logical_edge_id,
            projection::logical_edge_id_width_v1::u32};
        support_edges[index].source_index = edges[index].source_index;
        support_edges[index].destination_index =
            edges[index].destination_index;
    }
    std::vector<projection::projection_value_map_v1> aligned_map(edges.size());
    for (std::size_t index = 0; index < edges.size(); ++index) {
        aligned_map[index].logical_edge_id = support_edges[index].logical_edge_id;
        aligned_map[index].region_kind =
            projection::physical_region_kind_v1::residual;
        aligned_map[index].region_index = static_cast<std::uint32_t>(index);
        aligned_map[index].projection_slot = static_cast<std::uint32_t>(index);
    }
    const std::array<std::uint8_t, source_count> source_support{1u, 0u, 1u, 1u};
    const std::array<std::uint8_t, destination_count> destination_support{
        1u, 1u, 0u};
    std::vector<sm70::support_projection_edge_v1> selected(edges.size());
    sm70::contract_projection_result_v1 projection_result{};
    const sm70::contract_projection_request_v1 projection_request{
        support_edges.data(), aligned_map.data(), support_edges.size(),
        source_support.data(), source_count, destination_support.data(),
        destination_count, selected.data(), selected.size()};
    require(sm70::prepare_contract_projection_v1(
                projection_request, &projection_result)
            == sm70::contract_projection_status_v1::success,
        "support projection preparation failed");
    require(projection_result.selected_edge_count == 3u,
        "support projection selected wrong edge count");

    const std::vector<__half> source_half = to_half(std::vector<float>(
        source.begin(), source.end()));
    const std::vector<__half> destination_features_half = to_half({
        0.5f, 0.25f, -0.75f, 1.25f, -0.5f,
        1.0f, -0.25f, 0.5f, 0.125f, 0.75f,
        -0.5f, 1.5f, 0.25f, -1.0f, 0.375f});
    device_buffer<sm70::support_logical_edge_v1> d_support_edges(edges.size());
    d_support_edges.upload(support_edges);
    selected.resize(projection_result.selected_edge_count);
    device_buffer<sm70::support_projection_edge_v1> d_selected(selected.size());
    d_selected.upload(selected);
    device_buffer<__half> d_source(source_half.size());
    d_source.upload(source_half);
    device_buffer<__half> d_destination_features(destination_features_half.size());
    d_destination_features.upload(destination_features_half);
    device_buffer<float> d_contraction(edges.size());
    const sm70::contract_on_support_request_v1 contract_request{
        d_support_edges.data(), edges.size(), d_selected.data(), selected.size(),
        d_source.data(), source_count, d_destination_features.data(),
        destination_count, width, d_contraction.data(), stream};
    require(sm70::enqueue_contract_on_support_v1(contract_request)
            == sm70::contract_on_support_status_v1::success,
        "support contraction launch failed");
    require_cuda(cudaStreamSynchronize(stream), "support synchronize");
    const auto contraction = d_contraction.download();
    for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
        const auto &edge = edges[edge_index];
        double expected = 0.0;
        if (source_support[edge.source_index]
            && destination_support[edge.destination_index])
            for (std::uint32_t column = 0u; column < width; ++column)
                expected += half_value(source_half[
                                edge.source_index * width + column])
                    * half_value(destination_features_half[
                        edge.destination_index * width + column]);
        require_close(contraction[edge_index], expected, 2.0e-5,
            "support contraction " + std::to_string(edge_index));
    }

    const std::array<std::size_t, 5u> permutation{3u, 0u, 4u, 1u, 2u};
    std::vector<projection::projection_value_map_v1> permuted_map(edges.size());
    for (std::size_t physical = 0; physical < edges.size(); ++physical) {
        permuted_map[physical] = aligned_map[permutation[physical]];
        permuted_map[physical].projection_slot =
            static_cast<std::uint32_t>(physical);
    }
    device_buffer<projection::projection_value_map_v1> d_permuted_map(
        permuted_map.size());
    d_permuted_map.upload(permuted_map);
    device_buffer<float> d_edge_gradient(edges.size());
    const sm70::edge_value_gradient_request_v1 edge_gradient_request{
        d_support_edges.data(), d_permuted_map.data(), edges.size(),
        d_source.data(), source_count, d_destination_gradient.data(),
        destination_count, width, d_edge_gradient.data(), stream};
    require(sm70::enqueue_edge_value_gradient_v1(edge_gradient_request)
            == sm70::edge_value_gradient_status_v1::success,
        "logical edge gradient launch failed");
    require_cuda(cudaStreamSynchronize(stream), "edge gradient synchronize");
    const auto edge_gradient = d_edge_gradient.download();
    std::vector<double> half_source(source_half.size());
    std::transform(source_half.begin(), source_half.end(), half_source.begin(),
        half_value);
    for (std::size_t index = 0; index < edges.size(); ++index) {
        auto plus = edge_values;
        auto minus = edge_values;
        plus[index] += epsilon;
        minus[index] -= epsilon;
        const double finite_difference =
            (relation_objective(edges, plus, half_source,
                 destination_gradient, width)
                - relation_objective(edges, minus, half_source,
                    destination_gradient, width))
            / (2.0 * epsilon);
        require_close(edge_gradient[index], finite_difference, 2.0e-5,
            "logical edge gradient " + std::to_string(index));
    }
}

double softmax_objective(const std::vector<double> &values,
    const std::vector<double> &gradient,
    const std::array<std::uint64_t, 3u> &offsets, std::uint32_t width) {
    double objective = 0.0;
    for (std::size_t segment_index = 0u; segment_index + 1u < offsets.size();
         ++segment_index)
        for (std::uint32_t column = 0u; column < width; ++column) {
            double maximum = values[offsets[segment_index] * width + column];
            for (std::uint64_t row = offsets[segment_index] + 1u;
                 row < offsets[segment_index + 1u]; ++row)
                maximum = std::max(maximum, values[row * width + column]);
            double denominator = 0.0;
            for (std::uint64_t row = offsets[segment_index];
                 row < offsets[segment_index + 1u]; ++row)
                denominator += std::exp(values[row * width + column] - maximum);
            for (std::uint64_t row = offsets[segment_index];
                 row < offsets[segment_index + 1u]; ++row)
                objective += std::exp(values[row * width + column] - maximum)
                    / denominator * gradient[row * width + column];
        }
    return objective;
}

void validate_segment_softmax(cudaStream_t stream) {
    constexpr std::uint64_t value_count = 5u;
    constexpr std::uint32_t segment_count = 2u;
    constexpr std::uint32_t width = 3u;
    const std::array<std::uint64_t, 3u> offsets{0u, 3u, 5u};
    const std::vector<float> values{0.25f, -0.5f, 1.0f, 1.25f, 0.75f,
        -0.25f, -0.75f, 1.5f, 0.5f, 0.625f, -1.0f, 1.25f, -0.125f,
        0.375f, -0.625f};
    const std::vector<float> output_gradient{0.5f, -0.75f, 1.25f, -1.0f,
        0.25f, 0.5f, 1.5f, -0.125f, -0.5f, 0.75f, 0.625f, -1.25f,
        -0.375f, 1.0f, 0.875f};
    device_buffer<std::uint64_t> d_offsets(offsets.size());
    d_offsets.upload(std::vector<std::uint64_t>(offsets.begin(), offsets.end()));
    device_buffer<float> d_values(values.size());
    d_values.upload(values);
    device_buffer<float> d_normalized(values.size());
    device_buffer<float> d_output_gradient(output_gradient.size());
    d_output_gradient.upload(output_gradient);
    device_buffer<float> d_input_gradient(values.size());

    const auto values_axis = make_axis(1u);
    const auto segment_axis = make_axis(101u);
    const auto dense_axis = make_axis(201u);
    const execution::device_location location{
        execution::residency_kind::device, {}, 0, 0u};
    segment::segment_normalize_plan_v1 plan{};
    plan.kind = segment::segment_normalize_kind_v1::softmax;
    plan.values_axis = values_axis;
    plan.segment_axis = segment_axis;
    plan.dense_axis = dense_axis;
    plan.value_count = value_count;
    plan.segment_count = segment_count;
    plan.dense_width = width;
    const segment::segment_partition_view_v1 partition{values_axis,
        segment_axis, d_offsets.data(), location, value_count, segment_count,
        static_cast<std::uint32_t>(offsets.size())};
    const execution::stream_context stream_context{
        reinterpret_cast<void *>(stream), 0, 0u};
    const execution::transient_workspace workspace{nullptr, 0u, location};
    const auto input_view = make_matrix(d_values.data(), location, values_axis,
        dense_axis, value_count, width);
    const auto normalized_view = make_matrix(d_normalized.data(), location,
        values_axis, dense_axis, value_count, width);
    const auto output_gradient_view = make_matrix(d_output_gradient.data(),
        location, values_axis, dense_axis, value_count, width);
    const auto input_gradient_view = make_matrix(d_input_gradient.data(),
        location, values_axis, dense_axis, value_count, width);
    require(static_cast<bool>(segment::run_segment_softmax_v1(plan, partition,
                input_view, normalized_view, stream_context, workspace)),
        "segment softmax forward failed");
    const sm70::prepared_segment_backward_request_v1 backward_request{&plan,
        &partition, input_view, normalized_view, output_gradient_view,
        input_gradient_view, stream_context, workspace};
    require(sm70::enqueue_prepared_segment_backward_v1(backward_request)
            == sm70::prepared_segment_backward_status_v1::success,
        "prepared segment softmax backward failed");
    require_cuda(cudaStreamSynchronize(stream), "segment synchronize");
    const auto input_gradient = d_input_gradient.download();
    const std::vector<double> double_values(values.begin(), values.end());
    const std::vector<double> double_gradient(
        output_gradient.begin(), output_gradient.end());
    constexpr double epsilon = 1.0e-5;
    for (std::size_t index = 0; index < values.size(); ++index) {
        auto plus = double_values;
        auto minus = double_values;
        plus[index] += epsilon;
        minus[index] -= epsilon;
        const double finite_difference =
            (softmax_objective(plus, double_gradient, offsets, width)
                - softmax_objective(minus, double_gradient, offsets, width))
            / (2.0 * epsilon);
        require_close(input_gradient[index], finite_difference, 3.0e-5,
            "segment softmax gradient " + std::to_string(index));
    }
}

struct exchange_state {
    double input = 0.0;
    double edge_value = 0.0;
    double gate = 0.0;
    double competing_logit = 0.0;
    double relation_value = 0.0;
    double contracted = 0.0;
    double gated = 0.0;
    double normalized = 0.0;
    double output = 0.0;
    double input_gradient = 0.0;
    std::uint32_t next_step = 0u;
    void *expected_stream = nullptr;
};

bool validate_exchange_call(exchange_state *state, void *stream,
    std::uint32_t step) noexcept {
    if (state == nullptr || stream != state->expected_stream
        || state->next_step != step)
        return false;
    ++state->next_step;
    return true;
}

bool exchange_contract(void *context, void *stream, std::uint64_t input,
    std::uint64_t output) noexcept {
    auto *state = static_cast<exchange_state *>(context);
    if (!validate_exchange_call(state, stream, 0u) || input != 1u
        || output != 2u)
        return false;
    state->contracted = state->input * state->edge_value;
    return true;
}

bool exchange_gate(void *context, void *stream, std::uint64_t input,
    std::uint64_t output) noexcept {
    auto *state = static_cast<exchange_state *>(context);
    if (!validate_exchange_call(state, stream, 1u) || input != 2u
        || output != 3u)
        return false;
    state->gated = state->contracted * state->gate;
    return true;
}

bool exchange_normalize(void *context, void *stream, std::uint64_t input,
    std::uint64_t output) noexcept {
    auto *state = static_cast<exchange_state *>(context);
    if (!validate_exchange_call(state, stream, 2u) || input != 3u
        || output != 4u)
        return false;
    const double first = std::exp(state->gated);
    const double second = std::exp(state->competing_logit);
    state->normalized = first / (first + second);
    return true;
}

bool exchange_relation(void *context, void *stream, std::uint64_t input,
    std::uint64_t output) noexcept {
    auto *state = static_cast<exchange_state *>(context);
    if (!validate_exchange_call(state, stream, 3u) || input != 4u
        || output != 5u)
        return false;
    state->output = state->relation_value * state->normalized;
    state->input_gradient = state->relation_value * state->normalized
        * (1.0 - state->normalized) * state->gate * state->edge_value;
    return true;
}

double exchange_reference(const exchange_state &state, double input) {
    const double gated = input * state.edge_value * state.gate;
    return state.relation_value * std::exp(gated)
        / (std::exp(gated) + std::exp(state.competing_logit));
}

void validate_composed_exchange() {
    int stream_token = 0;
    exchange_state state{0.375, -1.25, 0.625, -0.4, 1.75, 0.0, 0.0,
        0.0, 0.0, 0.0, 0u, &stream_token};
    using operation = cellerator::compute::operation::relation_algebra_kind_v1;
    const std::array<sm70::prepared_exchange_step_v1, 4u> steps{{
        {operation::contract_on_support, exchange_contract, &state, 1u, 2u},
        {operation::edge_map_or_gate, exchange_gate, &state, 2u, 3u},
        {operation::segment_normalize, exchange_normalize, &state, 3u, 4u},
        {operation::relation_apply, exchange_relation, &state, 4u, 5u}}};
    const sm70::prepared_exchange_program_v1 program{
        1u, static_cast<std::uint32_t>(steps.size()), steps.data(),
        &stream_token};
    require(sm70::run_prepared_exchange_program_v1(program)
            == sm70::exchange_program_status_v1::success,
        "composed exchange program failed");
    require(state.next_step == steps.size(),
        "composed exchange did not execute every step once in order");
    constexpr double epsilon = 1.0e-6;
    const double finite_difference =
        (exchange_reference(state, state.input + epsilon)
            - exchange_reference(state, state.input - epsilon))
        / (2.0 * epsilon);
    require_close(state.input_gradient, finite_difference, 1.0e-9,
        "composed exchange gradient");
}

} // namespace

int main() {
    int device_count = 0;
    require_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    require(device_count > 0, "CUDA device is required");
    require_cuda(cudaSetDevice(0), "cudaSetDevice");
    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    validate_transpose_and_edge_values(stream);
    validate_segment_softmax(stream);
    require_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    validate_composed_exchange();
    std::cout << "gradient validation passed\n";
    return EXIT_SUCCESS;
}
