#include <Cellerator/execution/joint_compiler/coverage_roles_v1.hh>

namespace cellerator::execution::joint_compiler {
namespace {

bool zero_identity(persistent_identity_v1 identity) noexcept {
    return identity.producer_namespace == 0u && identity.local_identity == 0u;
}

}  // namespace

coverage_role_validation_result_v1 validate_coverage_role_flags_v1(
    std::uint32_t role_flags) noexcept {
    if (role_flags == 0u)
        return {coverage_role_validation_code_v1::missing_role};
    if ((role_flags & ~known_coverage_role_flags_v1) != 0u)
        return {coverage_role_validation_code_v1::unknown_role};

    const bool proposal =
        (role_flags & approximate_proposal_membership_role_v1) != 0u;
    if (proposal) {
        if (role_flags != approximate_proposal_membership_role_v1)
            return {
                coverage_role_validation_code_v1::proposal_execution_mixture};
        return {};
    }
    if ((role_flags & certified_exact_coverage_role_v1) == 0u)
        return {
            coverage_role_validation_code_v1::missing_exact_certification};

    const bool read_requirement =
        (role_flags & exact_read_requirement_role_v1) != 0u;
    const bool halo = (role_flags & read_only_halo_role_v1) != 0u;
    const bool replica = (role_flags & physical_replica_role_v1) != 0u;
    const bool exclusive =
        (role_flags & exclusive_output_owner_role_v1) != 0u;
    const bool partial =
        (role_flags & partial_contribution_owner_role_v1) != 0u;

    if (halo && !read_requirement)
        return {
            coverage_role_validation_code_v1::halo_without_read_requirement};
    if ((halo || replica) && (exclusive || partial))
        return {
            coverage_role_validation_code_v1::read_only_role_writes_output};
    if (exclusive && partial)
        return {
            coverage_role_validation_code_v1::ambiguous_output_ownership};
    return {};
}

coverage_role_validation_result_v1 validate_coverage_role_record_v1(
    const coverage_role_record_v1 &record) noexcept {
    if (record.schema_version != coverage_role_schema_version_v1)
        return {coverage_role_validation_code_v1::unsupported_schema};
    if (record.record_bytes != sizeof(coverage_role_record_v1))
        return {coverage_role_validation_code_v1::invalid_record_bytes};
    if (record.reserved != 0u)
        return {coverage_role_validation_code_v1::nonzero_reserved};
    if (!validate_persistent_identity_v1(record.coverage_identity))
        return {coverage_role_validation_code_v1::invalid_coverage_identity};
    if (!validate_persistent_identity_v1(record.participant_identity))
        return {
            coverage_role_validation_code_v1::invalid_participant_identity};
    const coverage_role_validation_result_v1 flags =
        validate_coverage_role_flags_v1(record.role_flags);
    if (!flags) return flags;

    const bool partial =
        (record.role_flags & partial_contribution_owner_role_v1) != 0u;
    if (partial && !validate_persistent_identity_v1(
            record.partial_algebra_identity))
        return {coverage_role_validation_code_v1::missing_partial_algebra};
    if (!partial && !zero_identity(record.partial_algebra_identity))
        return {coverage_role_validation_code_v1::unexpected_partial_algebra};
    return {};
}

}  // namespace cellerator::execution::joint_compiler
