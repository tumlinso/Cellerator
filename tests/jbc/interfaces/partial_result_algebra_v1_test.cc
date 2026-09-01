#include <Cellerator/compute/decomposition/partial_result_algebra_v1.hh>

#include <cassert>
#include <cstdint>

namespace decomposition = cellerator::compute::decomposition;
namespace execution = cellerator::execution;
namespace operation = cellerator::compute::operation::v2;

decomposition::partial_result_algebra_v1 valid_algebra() {
    decomposition::partial_result_algebra_v1 algebra{};
    algebra.algebra_identity = {1u, 1u};
    algebra.state_layout_identity = {1u, 2u};
    algebra.neutral_element_identity = {1u, 3u};
    algebra.merge_operation_identity = {1u, 4u};
    algebra.finalize_operation_identity = {1u, 5u};
    algebra.state_bytes = 16u;
    algebra.state_alignment = 16u;
    algebra.flags = decomposition::associative_v1
        | decomposition::commutative_v1
        | decomposition::deterministic_tree_required_v1;
    algebra.deterministic_tree_identity = {2u, 1u};
    algebra.numerical.relation_storage = execution::numeric_type::f16;
    algebra.numerical.state_storage = execution::numeric_type::f32;
    algebra.numerical.multiply = execution::numeric_type::f32;
    algebra.numerical.accumulation = execution::numeric_type::f32;
    algebra.numerical.output_storage = execution::numeric_type::f32;
    algebra.numerical.scalar = execution::numeric_type::f32;
    return algebra;
}

int main() {
    auto algebra = valid_algebra();
    assert(decomposition::validate_partial_result_algebra_v1(algebra));

    auto malformed = algebra;
    malformed.flags = decomposition::commutative_v1;
    malformed.deterministic_tree_identity = {};
    assert(decomposition::validate_partial_result_algebra_v1(malformed).code
        == decomposition::partial_result_algebra_validation_code_v1::
            missing_reconstruction_rule);

    malformed = algebra;
    malformed.flags = decomposition::ordered_only_v1;
    malformed.deterministic_tree_identity = {};
    assert(decomposition::validate_partial_result_algebra_v1(malformed).code
        == decomposition::partial_result_algebra_validation_code_v1::
            invalid_order_constraint);
    malformed.required_merge_order = {3u, 1u};
    assert(decomposition::validate_partial_result_algebra_v1(malformed));

    malformed = algebra;
    malformed.flags = decomposition::associative_v1;
    assert(decomposition::validate_partial_result_algebra_v1(malformed).code
        == decomposition::partial_result_algebra_validation_code_v1::
            unexpected_deterministic_tree);
    malformed.deterministic_tree_identity = {};
    assert(decomposition::validate_partial_result_algebra_v1(malformed));

    malformed = algebra;
    malformed.state_alignment = 24u;
    assert(decomposition::validate_partial_result_algebra_v1(malformed).code
        == decomposition::partial_result_algebra_validation_code_v1::
            invalid_state_alignment);
    malformed = algebra;
    malformed.numerical.accumulation = execution::numeric_type::invalid;
    assert(decomposition::validate_partial_result_algebra_v1(malformed).code
        == decomposition::partial_result_algebra_validation_code_v1::
            invalid_numerical_policy);

    // All property combinations are classified without invoking providers.
    for (std::uint32_t flags = 0u;
         flags <= decomposition::known_partial_result_algebra_flags_v1;
         ++flags) {
        auto candidate = algebra;
        candidate.flags = flags;
        candidate.required_merge_order =
            (flags & decomposition::ordered_only_v1) != 0u
                ? execution::order_id{3u, 1u} : execution::order_id{};
        candidate.deterministic_tree_identity =
            (flags & decomposition::deterministic_tree_required_v1) != 0u
                ? execution::joint_compiler::persistent_identity_v1{2u, 1u}
                : execution::joint_compiler::persistent_identity_v1{};
        const auto result =
            decomposition::validate_partial_result_algebra_v1(candidate);
        if (result)
            assert((flags & (decomposition::associative_v1
                | decomposition::ordered_only_v1)) != 0u);
    }
    return 0;
}
