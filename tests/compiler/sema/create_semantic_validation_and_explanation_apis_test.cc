#include <Cellerator/compiler/sema/create_semantic_validation_and_explanation_apis_v1.hh>

#include <array>
#include <cassert>
#include <string>

int main() {
    using namespace cellerator::compiler::sema::v1;
    constexpr std::array<semantic_mismatch, 7> kinds{{
        semantic_mismatch::domain, semantic_mismatch::order,
        semantic_mismatch::structure_generation, semantic_mismatch::value_generation,
        semantic_mismatch::support_generation, semantic_mismatch::numerical_policy,
        semantic_mismatch::operation_resolution}};
    for (const auto kind : kinds) {
        const auto result = explain_semantic_compatibility(kind, "operand", "A", "B");
        assert(!result);
        assert(result.diagnostic == std::string(semantic_mismatch_name(kind))
            + " mismatch for operand: expected A, got B");
    }
}
