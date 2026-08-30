#include <Cellerator/compute/candidate/segment/normalize.hh>

#include <cstdint>

namespace cellerator::compute::architecture::providers::nvidia::sm70 {

enum class prepared_segment_backward_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument = 1u,
    segment_failure = 2u
};

// This is a prepared-program binding only. Segment owns the numerical
// primitive; the relation program owns sequencing and explicit operands. No
// framework tape, allocation, stream, or generic autograd object is created.
struct prepared_segment_backward_request_v1 {
    const compute::segment::segment_normalize_plan_v1 *plan = nullptr;
    const compute::segment::segment_partition_view_v1 *partition = nullptr;
    execution::dense_tensor_view input{};
    execution::dense_tensor_view forward_output{};
    execution::dense_tensor_view output_gradient{};
    execution::dense_tensor_view input_gradient{};
    execution::stream_context stream{};
    execution::transient_workspace workspace{};
};

prepared_segment_backward_status_v1 enqueue_prepared_segment_backward_v1(
    const prepared_segment_backward_request_v1 &request) noexcept {
    if (request.plan == nullptr || request.partition == nullptr)
        return prepared_segment_backward_status_v1::invalid_argument;
    compute::segment::segment_normalize_result_v1 result{};
    if (request.plan->kind
        == compute::segment::segment_normalize_kind_v1::log_sum_exp) {
        result = compute::segment::run_segment_log_sum_exp_backward_v1(
            *request.plan, *request.partition, request.input,
            request.forward_output, request.output_gradient,
            request.input_gradient, request.stream, request.workspace);
    } else if (request.plan->kind
        == compute::segment::segment_normalize_kind_v1::softmax) {
        result = compute::segment::run_segment_softmax_backward_v1(
            *request.plan, *request.partition, request.forward_output,
            request.output_gradient, request.input_gradient, request.stream,
            request.workspace);
    } else {
        return prepared_segment_backward_status_v1::invalid_argument;
    }
    return result
        ? prepared_segment_backward_status_v1::success
        : prepared_segment_backward_status_v1::segment_failure;
}

} // namespace cellerator::compute::architecture::providers::nvidia::sm70
