#include <Cellerator/compiler/profile/represent_domain_axis_relation_and_support_evidence_v1.hh>

#include <cmath>
#include <limits>

namespace cellerator::compiler::profile::v1 {

structural_profile_evidence_status_v1 derive_exact_structural_profile_evidence_v1(
    const cellerator::geometry::support_relation_view_v1 &relation,
    profile_identity_v1 evidence_identity, double confidence,
    structural_profile_evidence_v1 *evidence) noexcept {
    if (evidence == nullptr || (relation.destination_count != 0u
                                && relation.destination_offsets == nullptr)
        || (relation.edge_count != 0u && relation.source_count == 0u))
        return structural_profile_evidence_status_v1::invalid_argument;
    if (!std::isfinite(confidence) || confidence < 0.0 || confidence > 1.0)
        return structural_profile_evidence_status_v1::invalid_confidence;
    if (relation.destination_count != 0u && relation.destination_offsets[0] != 0u)
        return structural_profile_evidence_status_v1::invalid_offsets;
    structural_profile_evidence_v1 result{};
    result.evidence = evidence_identity;
    result.relation = {relation.relation_identity, 0u};
    result.structure = {relation.structure_identity, 0u};
    result.structure_epoch = relation.structure_epoch;
    result.source_axis.axis = {relation.source_axis_identity, 0u};
    result.source_axis.extent = relation.source_count;
    result.destination_axis.axis = {relation.destination_axis_identity, 0u};
    result.destination_axis.extent = relation.destination_count;
    result.degree.observation_count = relation.destination_count;
    result.occupancy.observation_count = relation.destination_count;
    result.degree.minimum = relation.destination_count == 0u
        ? 0.0 : std::numeric_limits<double>::infinity();
    double degree_sum = 0.0;
    double degree_square_sum = 0.0;
    for (std::uint32_t i = 0; i < relation.destination_count; ++i) {
        const auto begin = relation.destination_offsets[i];
        const auto end = relation.destination_offsets[i + 1u];
        if (end < begin || end > relation.edge_count)
            return structural_profile_evidence_status_v1::invalid_offsets;
        const auto degree = static_cast<double>(end - begin);
        result.degree.minimum = degree < result.degree.minimum ? degree : result.degree.minimum;
        result.degree.maximum = degree > result.degree.maximum ? degree : result.degree.maximum;
        degree_sum += degree;
        degree_square_sum += degree * degree;
        if (degree != 0.0) ++result.nonempty_destination_count;
    }
    if (relation.destination_count != 0u
        && relation.destination_offsets[relation.destination_count] != relation.edge_count)
        return structural_profile_evidence_status_v1::invalid_offsets;
    result.support_count = relation.edge_count;
    if (relation.destination_count != 0u) {
        result.degree.mean = degree_sum / relation.destination_count;
        result.degree.second_moment = degree_square_sum / relation.destination_count;
        result.occupancy.minimum = result.degree.minimum / relation.source_count;
        result.occupancy.maximum = result.degree.maximum / relation.source_count;
        result.occupancy.mean = result.degree.mean / relation.source_count;
        result.occupancy.second_moment = result.degree.second_moment
            / (static_cast<double>(relation.source_count) * relation.source_count);
    }
    result.confidence = confidence;
    *evidence = result;
    return structural_profile_evidence_status_v1::ok;
}

structural_profile_evidence_status_v1 validate_structural_profile_evidence_v1(
    const structural_profile_evidence_v1 &evidence) noexcept {
    if (evidence.schema_version != structural_profile_evidence_schema_version_v1)
        return structural_profile_evidence_status_v1::unsupported_schema;
    if (!std::isfinite(evidence.confidence) || evidence.confidence < 0.0
        || evidence.confidence > 1.0 || !std::isfinite(evidence.ordering_stability)
        || evidence.ordering_stability < 0.0 || evidence.ordering_stability > 1.0)
        return structural_profile_evidence_status_v1::invalid_confidence;
    if (evidence.support_count != static_cast<std::uint64_t>(evidence.degree.mean
            * evidence.degree.observation_count))
        return structural_profile_evidence_status_v1::identity_mismatch;
    return structural_profile_evidence_status_v1::ok;
}
}  // namespace cellerator::compiler::profile::v1
