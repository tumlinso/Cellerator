#include <Cellerator/compiler/pass/implement_pass_manager_and_analysis_invalidation_v1.hh>

#include <chrono>

namespace cellerator::compiler::pass::v1 {
namespace {
std::uint64_t mix(std::uint64_t hash, std::uint64_t value) {
    hash ^= value + UINT64_C(0x9e3779b97f4a7c15) + (hash << 6) + (hash >> 2);
    return hash;
}
}  // namespace

pass_pipeline_receipt_v1 run_pass_pipeline_v1(
    const std::vector<pass_descriptor_v1>& pipeline,
    pass_context_v1 context, pass_cancelled_v1 cancelled,
    void* cancellation_context) noexcept {
    pass_pipeline_receipt_v1 receipt{};
    receipt.deterministic_replay_hash = UINT64_C(0xcbf29ce484222325);
    std::uint16_t previous_stage = 0;
    bool first = true;
    for (const auto& descriptor : pipeline) {
        const auto stage = stable_stage_id_v1(descriptor.stage);
        if (descriptor.name.empty() || descriptor.run == nullptr
            || !valid_pipeline_stage_v1(descriptor.stage)
            || (!first && stage < previous_stage)) {
            receipt.status = pass_status_v1::invalid_pipeline;
            return receipt;
        }
        first = false;
        previous_stage = stage;
        if (cancelled != nullptr && cancelled(cancellation_context)) {
            receipt.status = pass_status_v1::cancelled;
            return receipt;
        }
        if ((context.available_analyses & descriptor.required_analyses)
            != descriptor.required_analyses) {
            receipt.status = pass_status_v1::missing_required_analysis;
            receipt.diagnostics.push_back(descriptor.name
                + ": required analysis unavailable");
            return receipt;
        }
        context.scope_depth = descriptor.scope_depth;
        pass_result_v1 result{};
        const auto begin = std::chrono::steady_clock::now();
        const bool success = descriptor.run(context, result);
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin).count();
        receipt.ordered_passes.push_back(descriptor.name);
        receipt.timings.push_back(
            {descriptor.name, static_cast<std::uint64_t>(elapsed)});
        if (!result.diagnostic.empty())
            receipt.diagnostics.push_back(descriptor.name + ": " + result.diagnostic);
        receipt.deterministic_replay_hash = mix(
            receipt.deterministic_replay_hash, stage);
        for (char character : descriptor.name)
            receipt.deterministic_replay_hash = mix(
                receipt.deterministic_replay_hash,
                static_cast<unsigned char>(character));
        if (!success) {
            receipt.status = pass_status_v1::pass_failed;
            return receipt;
        }
        if (result.changed) {
            ++context.module_revision;
            context.available_analyses &= result.preserved_analyses;
        }
        context.available_analyses |= result.produced_analyses;
        receipt.deterministic_replay_hash = mix(
            receipt.deterministic_replay_hash, context.module_revision);
        receipt.deterministic_replay_hash = mix(
            receipt.deterministic_replay_hash, context.available_analyses);
    }
    receipt.final_analyses = context.available_analyses;
    return receipt;
}

}  // namespace cellerator::compiler::pass::v1
