#include <Cellerator/compute/operation/fusion/fusion_validation_v1.hh>

#include <algorithm>
#include <cmath>
#include <limits>

namespace cellerator::compute::operation::fusion {
namespace {

constexpr std::uint64_t fnv1a(const char *text,
    std::uint64_t value = 1469598103934665603ull) noexcept {
    return *text == '\0' ? value
        : fnv1a(text + 1,
            (value ^ static_cast<std::uint8_t>(*text)) * 1099511628211ull);
}

#define CE_FUSION_ENTRY(name, kind, fused_value) \
    {fnv1a(name), name, composition_kind_v1::kind, fused_value, true, true, \
        true, false, true, true}

constexpr registry_entry_v1 entries[] = {
    CE_FUSION_ENTRY("fusion.value-generation-pack.unfused.v1",
        value_generation_to_pack, false),
    CE_FUSION_ENTRY("fusion.value-generation-pack.fused.v1",
        value_generation_to_pack, true),
    CE_FUSION_ENTRY("fusion.value-pack-apply.unfused.v1",
        value_pack_to_relation_apply, false),
    CE_FUSION_ENTRY("fusion.value-pack-apply.fused.v1",
        value_pack_to_relation_apply, true),
    CE_FUSION_ENTRY("fusion.mma-residual.unfused.v1",
        mma_to_same_owner_residual, false),
    CE_FUSION_ENTRY("fusion.mma-residual.fused.v1",
        mma_to_same_owner_residual, true),
    CE_FUSION_ENTRY("fusion.apply-epilogue.unfused.v1",
        relation_apply_to_epilogue, false),
    CE_FUSION_ENTRY("fusion.apply-epilogue.fused.v1",
        relation_apply_to_epilogue, true),
    CE_FUSION_ENTRY("fusion.contraction-edge-map.unfused.v1",
        contraction_to_edge_map, false),
    CE_FUSION_ENTRY("fusion.contraction-edge-map.fused.v1",
        contraction_to_edge_map, true),
    CE_FUSION_ENTRY("fusion.contraction-segment-statistic.unfused.v1",
        contraction_to_segment_statistic, false),
    CE_FUSION_ENTRY("fusion.contraction-segment-statistic.fused.v1",
        contraction_to_segment_statistic, true),
    CE_FUSION_ENTRY("fusion.normalize-apply.unfused.v1",
        normalization_to_relation_apply, false),
    CE_FUSION_ENTRY("fusion.normalize-apply.fused.v1",
        normalization_to_relation_apply, true),
    CE_FUSION_ENTRY("fusion.sparse-exchange.unfused.v1",
        sparse_exchange, false),
    CE_FUSION_ENTRY("fusion.sparse-exchange.fused.v1", sparse_exchange, true),
    CE_FUSION_ENTRY("fusion.bundle-shared-destination.unfused.v1",
        bundle_to_shared_destination, false),
    CE_FUSION_ENTRY("fusion.bundle-shared-destination.fused.v1",
        bundle_to_shared_destination, true),
    CE_FUSION_ENTRY("fusion.relation-moments.unfused.v1",
        relation_moments_pair, false),
    CE_FUSION_ENTRY("fusion.relation-moments.fused.v1",
        relation_moments_pair, true),
};

#undef CE_FUSION_ENTRY

constexpr std::size_t entry_count = sizeof(entries) / sizeof(entries[0]);

} // namespace

const registry_entry_v1 *fusion_registry_v1(std::size_t *count) noexcept {
    if (count != nullptr) *count = entry_count;
    return entries;
}

status_v1 validate_fusion_registry_v1() noexcept {
    for (std::size_t index = 0u; index < entry_count; ++index) {
        const registry_entry_v1 &entry = entries[index];
        if (entry.stable_candidate_id == 0u || entry.unique_name == nullptr
            || entry.composition > composition_kind_v1::relation_moments_pair
            || !entry.experimental || !entry.requires_measurement
            || !entry.explicitly_selectable || entry.auto_promoted
            || !entry.unfused_stages_available || !entry.exact)
            return status_v1::invalid_argument;
        for (std::size_t prior = 0u; prior < index; ++prior)
            if (entries[prior].stable_candidate_id
                == entry.stable_candidate_id)
                return status_v1::invalid_identity;
    }
    for (std::uint8_t composition = 0u;
        composition <= static_cast<std::uint8_t>(
            composition_kind_v1::relation_moments_pair); ++composition) {
        bool has_fused = false;
        bool has_unfused = false;
        for (const registry_entry_v1 &entry : entries)
            if (static_cast<std::uint8_t>(entry.composition) == composition) {
                has_fused = has_fused || entry.fused;
                has_unfused = has_unfused || !entry.fused;
            }
        if (!has_fused || !has_unfused) return status_v1::invalid_dependency;
    }
    return status_v1::success;
}

status_v1 validate_fused_unfused_equivalence_v1(
    const equivalence_request_v1 &request,
    equivalence_result_v1 *result) noexcept {
    if (result == nullptr || request.unfused_output == nullptr
        || request.fused_output == nullptr || request.local_output_count == 0u
        || !std::isfinite(request.absolute_tolerance)
        || !std::isfinite(request.relative_tolerance)
        || request.absolute_tolerance < 0.0
        || request.relative_tolerance < 0.0
        || request.global_output_begin
            > std::numeric_limits<std::uint64_t>::max()
                - request.local_output_count)
        return status_v1::invalid_argument;
    equivalence_result_v1 checked{};
    checked.exact_match = true;
    checked.within_tolerance = true;
    checked.first_failing_global_output =
        std::numeric_limits<std::uint64_t>::max();
    for (std::uint32_t index = 0u; index < request.local_output_count;
        ++index) {
        const double unfused = request.unfused_output[index];
        const double fused = request.fused_output[index];
        if (!std::isfinite(unfused) || !std::isfinite(fused))
            return status_v1::invalid_argument;
        const double absolute = std::abs(fused - unfused);
        const double relative = absolute / std::max(std::abs(unfused),
            std::numeric_limits<double>::min());
        checked.maximum_absolute_error =
            std::max(checked.maximum_absolute_error, absolute);
        checked.maximum_relative_error =
            std::max(checked.maximum_relative_error, relative);
        checked.exact_match = checked.exact_match && absolute == 0.0;
        const bool within = absolute <= request.absolute_tolerance
            + request.relative_tolerance * std::abs(unfused);
        if (!within && checked.within_tolerance)
            checked.first_failing_global_output =
                request.global_output_begin + index;
        checked.within_tolerance = checked.within_tolerance && within;
        ++checked.checked_output_count;
    }
    *result = checked;
    return status_v1::success;
}

} // namespace cellerator::compute::operation::fusion
