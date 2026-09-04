#include <Cellerator/compiler/backend/benchmark_cpu_backend_complete_cost_v1.hh>

#include <chrono>

namespace cellerator::compiler::backend::v1 {
namespace {

std::uint64_t measure(cpu_benchmark_step_v1 step, void* context,
    std::uint32_t count, bool* passed) {
    const auto begin = std::chrono::steady_clock::now();
    for (std::uint32_t i = 0; i < count; ++i) {
        if (!step(context)) {
            *passed = false;
            return 0;
        }
    }
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin).count());
}

}  // namespace

cpu_backend_cost_receipt_v1 benchmark_cpu_backend_complete_cost_v1(
    const cpu_backend_benchmark_request_v1& request) noexcept {
    cpu_backend_cost_receipt_v1 result{};
    if (request.compile == nullptr || request.prepare == nullptr
        || request.pack == nullptr || request.execute == nullptr
        || request.plain_cpp_baseline == nullptr
        || request.semantic_referee == nullptr || request.repeat_count == 0)
        return result;
    result.generated_source_bytes = request.generated_source_bytes;
    result.object_bytes = request.object_bytes;
    result.repeat_count = request.repeat_count;
    bool passed = true;
    result.compile_nanoseconds = measure(request.compile, request.context, 1, &passed);
    result.preparation_nanoseconds = measure(request.prepare, request.context, 1, &passed);
    result.packing_nanoseconds = measure(request.pack, request.context, 1, &passed);
    result.execution_nanoseconds = measure(request.execute, request.context, 1, &passed);
    for (std::uint32_t i = 0; passed && i < request.warmup_count; ++i)
        passed = request.execute(request.context);
    result.warm_reuse_nanoseconds = measure(
        request.execute, request.context, request.repeat_count, &passed);
    result.plain_cpp_nanoseconds = measure(
        request.plain_cpp_baseline, request.context, request.repeat_count, &passed);
    result.semantic_referee_passed = passed
        && request.semantic_referee(request.context);
    if (!result.semantic_referee_passed) return result;
    result.evaluation = result.warm_reuse_nanoseconds
            <= result.plain_cpp_nanoseconds
        ? cpu_backend_evaluation_v1::evaluated_promoted
        : cpu_backend_evaluation_v1::evaluated_not_promoted;
    return result;
}

}  // namespace cellerator::compiler::backend::v1
