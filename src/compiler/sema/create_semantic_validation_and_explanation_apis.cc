#include <Cellerator/compiler/sema/create_semantic_validation_and_explanation_apis_v1.hh>

#include <utility>

namespace cellerator::compiler::sema::v1 {

const char *semantic_mismatch_name(semantic_mismatch mismatch) noexcept {
    switch (mismatch) {
    case semantic_mismatch::none: return "compatible";
    case semantic_mismatch::domain: return "domain";
    case semantic_mismatch::order: return "order";
    case semantic_mismatch::structure_generation: return "structure-generation";
    case semantic_mismatch::value_generation: return "value-generation";
    case semantic_mismatch::support_generation: return "support-generation";
    case semantic_mismatch::numerical_policy: return "numerical-policy";
    case semantic_mismatch::operation_resolution: return "operation-resolution";
    }
    return "unknown";
}

semantic_explanation explain_semantic_compatibility(
    semantic_mismatch mismatch,
    std::string subject,
    std::string expected,
    std::string actual) {
    semantic_explanation result{mismatch, std::move(subject), std::move(expected),
                                std::move(actual), {}};
    result.diagnostic = std::string(semantic_mismatch_name(mismatch)) + " mismatch for "
        + result.subject + ": expected " + result.expected + ", got " + result.actual;
    return result;
}

}  // namespace cellerator::compiler::sema::v1
