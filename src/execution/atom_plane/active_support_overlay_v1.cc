#include "Cellerator/execution/atom_plane/active_support_overlay_v1.hh"

#include <limits>

namespace cellerator::execution::atom_plane {
namespace {

active_support_overlay_status_v1 failure(
    active_support_overlay_code_v1 code,
    u64 subject = 0u,
    u32 word_index = 0u,
    relation_value_atom_plane_code_v1 relation_code =
        relation_value_atom_plane_code_v1::success) noexcept {
    return {code, relation_code, 0u, word_index, subject};
}

u32 count_bits(u64 word) noexcept {
    u32 count = 0u;
    while (word != 0u) {
        word &= word - 1u;
        ++count;
    }
    return count;
}

}  // namespace

active_support_overlay_status_v1 validate_active_support_overlay_atom_plane_v1(
    const active_support_overlay_atom_plane_v1 &overlay,
    projection_value_plane::composite_validation_workspace_v1
        composite_workspace) noexcept {
    if (overlay.schema_version != active_support_overlay_schema_v1
        || overlay.reserved != 0u || overlay.relation_values == nullptr) {
        return failure(active_support_overlay_code_v1::invalid_argument);
    }
    if (!valid_external_atom_plane_identity_v1(overlay.plane_identity)) {
        return failure(
            active_support_overlay_code_v1::invalid_plane_identity);
    }
    const relation_value_atom_plane_status_v1 relation_status =
        validate_relation_value_atom_plane_v1(
            *overlay.relation_values, composite_workspace, nullptr);
    if (!relation_status) {
        return failure(active_support_overlay_code_v1::invalid_relation_values,
            relation_status.subject, 0u, relation_status.code);
    }
    if (overlay.relation_generation.value == 0u
        || overlay.relation_generation.value
            != overlay.relation_values->values->generation.value) {
        return failure(
            active_support_overlay_code_v1::stale_relation_generation,
            overlay.relation_generation.value);
    }
    if (overlay.overlay_generation.value == 0u) {
        return failure(
            active_support_overlay_code_v1::missing_overlay_generation);
    }
    if (!same_identity(overlay.logical_edge_order,
            overlay.relation_values->values->logical_edge_order)) {
        return failure(
            active_support_overlay_code_v1::logical_edge_order_mismatch);
    }
    if (!valid_location(overlay.location)) {
        return failure(active_support_overlay_code_v1::invalid_location);
    }
    const u64 logical_edge_count =
        overlay.relation_values->values->logical_edge_count;
    if (logical_edge_count > std::numeric_limits<u64>::max() - 63u) {
        return failure(active_support_overlay_code_v1::word_count_mismatch,
            logical_edge_count);
    }
    const u64 expected_words = (logical_edge_count + 63u) / 64u;
    if (overlay.word_count != expected_words) {
        return failure(active_support_overlay_code_v1::word_count_mismatch,
            overlay.word_count);
    }
    if (expected_words != 0u && overlay.active_words == nullptr) {
        return failure(active_support_overlay_code_v1::missing_active_words);
    }

    u64 observed_active = 0u;
    for (u64 index = 0u; index < expected_words; ++index) {
        const u64 word = overlay.active_words[index];
        if (index + 1u == expected_words
            && (logical_edge_count & 63u) != 0u) {
            const u32 valid_bits = static_cast<u32>(logical_edge_count & 63u);
            const u64 valid_mask = (u64{1u} << valid_bits) - 1u;
            if ((word & ~valid_mask) != 0u) {
                return failure(
                    active_support_overlay_code_v1::nonzero_tail_bits,
                    word & ~valid_mask, static_cast<u32>(index));
            }
        }
        observed_active += count_bits(word);
    }
    if (observed_active != overlay.active_edge_count) {
        return failure(
            active_support_overlay_code_v1::active_edge_count_mismatch,
            observed_active);
    }
    return {};
}

}  // namespace cellerator::execution::atom_plane
