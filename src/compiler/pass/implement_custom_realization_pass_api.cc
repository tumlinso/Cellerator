#include <Cellerator/compiler/pass/implement_custom_realization_pass_api_v1.hh>

#include <set>

namespace cellerator::compiler::pass::v1 {

realization_pass_status_v1 run_custom_realization_pass_v1(
    realization_pass_context_v1& context, realization_pass_run_v1 pass) noexcept {
    if (context.physical_covers == nullptr || context.projections == nullptr
        || context.packs == nullptr || context.stages == nullptr
        || context.bindings == nullptr || context.target_operations == nullptr
        || context.native_fragments == nullptr || context.diagnostics == nullptr
        || pass == nullptr)
        return realization_pass_status_v1::invalid_context;
    const auto original_stages = *context.stages;
    const auto original_bindings = *context.bindings;
    const auto original_fragments = *context.native_fragments;
    if (!pass(context)) {
        *context.stages = original_stages;
        *context.bindings = original_bindings;
        *context.native_fragments = original_fragments;
        return realization_pass_status_v1::pass_failed;
    }
    std::set<std::uint64_t> stages;
    for (const auto& stage : *context.stages) {
        if (stage.id == 0 || stage.operation.empty() || !stages.insert(stage.id).second)
            return realization_pass_status_v1::invalid_result;
        for (auto dependency : stage.dependencies)
            if (stages.count(dependency) == 0)
                return realization_pass_status_v1::invalid_result;
    }
    std::set<std::uint64_t> objects;
    for (const auto* collection : {context.physical_covers, context.projections,
             context.packs, context.target_operations})
        for (const auto& object : *collection)
            if (object.id == 0 || object.kind.empty() || !objects.insert(object.id).second)
                return realization_pass_status_v1::invalid_result;
    for (const auto& binding : *context.bindings)
        if (stages.count(binding.stage) == 0 || objects.count(binding.object) == 0)
            return realization_pass_status_v1::invalid_result;
    for (const auto& fragment : *context.native_fragments)
        if (stages.count(fragment.stage) == 0 || fragment.provider.empty()
            || fragment.bytes.empty())
            return realization_pass_status_v1::invalid_result;
    return realization_pass_status_v1::success;
}

}  // namespace cellerator::compiler::pass::v1
