#include "Cellerator/geometry/support_atlas.hh"

#include <cassert>
#include <cstdint>
#include <type_traits>

namespace geo = cellerator::geometry;

int main() {
    static_assert(std::is_trivially_copyable<geo::support_relation_view_v1>::value);
    static_assert(std::is_trivially_copyable<geo::support_sampling_policy_v1>::value);
    static_assert(std::is_trivially_copyable<geo::co_support_record_v1>::value);
    static_assert(std::is_trivially_copyable<geo::source_affinity_record_v1>::value);
    static_assert(std::is_trivially_copyable<geo::support_atlas_requirements_v1>::value);
    static_assert(std::is_trivially_copyable<geo::support_atlas_section_span_v1>::value);

    geo::support_atlas_view_v1 empty{};
    assert(empty.schema_version == geo::support_atlas_schema_version_v1);
    assert(empty.flags == geo::support_atlas_flag_none);
    assert(empty.prevalence == nullptr);
    assert(empty.prevalence_count == 0u);
    assert(empty.exact_rescans == nullptr);
    assert(empty.exact_rescan_count == 0u);

    geo::support_atlas_section_header_v1 persisted{};
    assert(persisted.schema_version == geo::support_atlas_section_schema_version_v1);
    assert(persisted.header_bytes == geo::support_atlas_section_header_bytes_v1);
    assert(persisted.section_bytes == 0u);
    assert(persisted.prevalence.byte_offset == 0u);
    assert(persisted.prevalence.element_count == 0u);

    geo::co_support_record_v1 pair{};
    pair.source_a = 3u;
    pair.source_b = 11u;
    pair.sampled_support = 7u;
    pair.association_numerator = 5;
    pair.association_denominator = 9u;
    assert(pair.source_a < pair.source_b);
    assert(pair.association_denominator != 0u);

    // Support evidence remains optional and hardware-neutral: its persisted
    // schema carries biological/structural identity and relative byte spans,
    // but no execution-provider or device identity.
    static_assert(sizeof(geo::support_atlas_section_span_v1) == 16u);
    return 0;
}
