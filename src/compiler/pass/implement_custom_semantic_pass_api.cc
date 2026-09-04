#include <Cellerator/compiler/pass/implement_custom_semantic_pass_api_v1.hh>

namespace cellerator::compiler::pass::v1 {

semantic_pass_status_v1 run_custom_semantic_pass_v1(
    semantic_pass_context_v1& context, semantic_pass_run_v1 run,
    semantic_pass_validate_v1 validate) noexcept {
    if (context.relation_applies == nullptr || context.profiles == nullptr
        || context.source_mappings == nullptr || context.diagnostics == nullptr
        || context.analysis_cache == nullptr || run == nullptr
        || validate == nullptr)
        return semantic_pass_status_v1::invalid_context;
    const auto original = *context.relation_applies;
    if (!run(context)) {
        *context.relation_applies = original;
        return semantic_pass_status_v1::pass_failed;
    }
    if (!validate(context)) {
        *context.relation_applies = original;
        context.diagnostics->push_back("custom semantic pass validation failed");
        return semantic_pass_status_v1::validation_failed;
    }
    return semantic_pass_status_v1::success;
}

}  // namespace cellerator::compiler::pass::v1
