#include <Cellerator/execution/joint_compiler/coverage_roles_v1.hh>

#include <cassert>
#include <cstdint>

namespace joint_compiler = cellerator::execution::joint_compiler;

int main() {
    using namespace joint_compiler;

    coverage_role_record_v1 role{};
    role.coverage_identity = {1u, 2u};
    role.participant_identity = {3u, 4u};
    role.role_flags = certified_exact_coverage_role_v1
        | exact_read_requirement_role_v1 | read_only_halo_role_v1;
    assert(validate_coverage_role_record_v1(role));

    role.role_flags = approximate_proposal_membership_role_v1;
    assert(validate_coverage_role_record_v1(role));
    role.role_flags |= certified_exact_coverage_role_v1;
    assert(validate_coverage_role_record_v1(role).code
        == coverage_role_validation_code_v1::proposal_execution_mixture);

    role.role_flags =
        certified_exact_coverage_role_v1 | read_only_halo_role_v1;
    assert(validate_coverage_role_record_v1(role).code
        == coverage_role_validation_code_v1::halo_without_read_requirement);

    role.role_flags = certified_exact_coverage_role_v1
        | exact_read_requirement_role_v1 | read_only_halo_role_v1
        | exclusive_output_owner_role_v1;
    assert(validate_coverage_role_record_v1(role).code
        == coverage_role_validation_code_v1::read_only_role_writes_output);

    role.role_flags = certified_exact_coverage_role_v1
        | exclusive_output_owner_role_v1
        | partial_contribution_owner_role_v1;
    assert(validate_coverage_role_record_v1(role).code
        == coverage_role_validation_code_v1::ambiguous_output_ownership);

    role.role_flags = certified_exact_coverage_role_v1
        | partial_contribution_owner_role_v1;
    assert(validate_coverage_role_record_v1(role).code
        == coverage_role_validation_code_v1::missing_partial_algebra);
    role.partial_algebra_identity = {5u, 6u};
    assert(validate_coverage_role_record_v1(role));

    role.role_flags = certified_exact_coverage_role_v1
        | exclusive_output_owner_role_v1;
    assert(validate_coverage_role_record_v1(role).code
        == coverage_role_validation_code_v1::unexpected_partial_algebra);
    role.partial_algebra_identity = {};
    assert(validate_coverage_role_record_v1(role));

    role.schema_version += 1u;
    assert(validate_coverage_role_record_v1(role).code
        == coverage_role_validation_code_v1::unsupported_schema);
    role.schema_version = coverage_role_schema_version_v1;
    role.record_bytes -= 1u;
    assert(validate_coverage_role_record_v1(role).code
        == coverage_role_validation_code_v1::invalid_record_bytes);

    // Exhaustively classify every seven-bit flag combination. Any accepted
    // execution role is exact; proposal membership is accepted only alone.
    for (std::uint32_t flags = 0u; flags <= known_coverage_role_flags_v1;
         ++flags) {
        const auto result = validate_coverage_role_flags_v1(flags);
        if (!result) continue;
        const bool proposal =
            (flags & approximate_proposal_membership_role_v1) != 0u;
        assert(proposal
            ? flags == approximate_proposal_membership_role_v1
            : (flags & certified_exact_coverage_role_v1) != 0u);
    }

    return 0;
}
