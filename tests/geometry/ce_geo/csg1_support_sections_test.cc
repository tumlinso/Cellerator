#include <Cellerator/geometry/persistence/semantic_geometry_image_v1.hh>
#include <Cellerator/geometry/support_atlas.hh>

#include <cassert>
#include <cstdint>
#include <cstring>

namespace cellerator::geometry::persistence {

inline constexpr u32 semantic_geometry_support_atlas_section_kind_v1 =
    semantic_geometry_first_optional_section_kind_v1;
inline constexpr u32 semantic_geometry_support_reference_section_kind_v1 =
    semantic_geometry_first_optional_section_kind_v1 + 1u;
inline constexpr u32 support_atlas_reference_schema_version_v1 = 1u;
enum class support_section_status_v1 : u8 {
    success = 0u, invalid_argument = 1u, invalid_atlas = 2u,
    arithmetic_overflow = 3u, insufficient_capacity = 4u,
    invalid_section = 5u
};
struct support_atlas_section_requirements_v1 {
    u64 section_bytes = 0u;
    u64 alignment = semantic_geometry_image_alignment_v1;
};
struct support_atlas_external_reference_v1 {
    u32 schema_version = support_atlas_reference_schema_version_v1;
    u32 record_bytes = sizeof(support_atlas_external_reference_v1);
    u64 evidence_identity = 0u;
    u64 relation_identity = 0u;
    u64 structure_identity = 0u;
    u64 structure_epoch = 0u;
    u64 source_axis_identity = 0u;
    u64 destination_axis_identity = 0u;
    u64 object_identity_low = 0u;
    u64 object_identity_high = 0u;
    u64 content_identity = 0u;
    u64 byte_offset = 0u;
    u64 byte_count = 0u;
    u64 reserved[3]{};
};
support_section_status_v1 query_support_atlas_section_requirements_v1(
    const support_atlas_view_v1 &,
    support_atlas_section_requirements_v1 *) noexcept;
support_section_status_v1 build_support_atlas_optional_section_v1(
    const support_atlas_view_v1 &, void *, u64,
    semantic_geometry_optional_section_v1 *) noexcept;
support_section_status_v1 rebind_support_atlas_section_v1(
    const semantic_geometry_section_view_v1 &,
    support_atlas_view_v1 *) noexcept;
support_section_status_v1 make_support_atlas_reference_section_v1(
    const support_atlas_external_reference_v1 &,
    semantic_geometry_optional_section_v1 *) noexcept;

} // namespace cellerator::geometry::persistence

namespace {

namespace geo = cellerator::geometry;
namespace persist = cellerator::geometry::persistence;
namespace ex = cellerator::execution;

ex::axis_identity compact_axis(std::uint32_t seed) {
    return {{seed + 1u, 1u}, {seed + 2u, 1u},
        {seed + 3u, 1u}, {seed + 4u, 1u}};
}

ex::persistent_axis_identity persistent_axis(std::uint64_t seed) {
    ex::persistent_axis_identity result{};
    result.header = {ex::biological_abi_version,
        ex::serialized_record_kind::persistent_axis_identity, sizeof(result)};
    result.domain = {seed + 1u, seed + 2u};
    result.order = {seed + 3u, seed + 4u};
    result.geometry = {seed + 5u, seed + 6u};
    result.partition = {seed + 7u, seed + 8u};
    return result;
}

struct fixture {
    geo::source_prevalence_v1 prevalence[2]{{0u, 0u, 3u, 2.5},
        {1u, 0u, 2u, 1.5}};
    geo::community_assignment_v1 communities[2]{{0u, 0u, 7u, 0u},
        {0u, 1u, 7u, 0u}};
    geo::exact_rescan_summary_v1 exact{77u, 5u, 5u, 0u};
    geo::support_atlas_view_v1 atlas{};
    std::uint32_t member = 0u;
    std::uint32_t permutation = 0u;
    geo::semantic_component_v1 component{
        1u, geo::semantic_component_kind::rectangular, {}, 0u, 1u};
    std::uint64_t edge = 0u;
    persist::semantic_geometry_image_build_request_v1 request{};

    fixture() {
        atlas.flags = geo::support_atlas_flag_sampled
            | geo::support_atlas_flag_multiresolution
            | geo::support_atlas_flag_exact_rescan;
        atlas.evidence_identity = 100u;
        atlas.relation_identity = 200u;
        atlas.structure_identity = 300u;
        atlas.structure_epoch = 4u;
        atlas.source_axis_identity = 400u;
        atlas.destination_axis_identity = 500u;
        atlas.source_count = 2u;
        atlas.destination_count = 1u;
        atlas.provenance.seed = 9u;
        atlas.provenance.input_identity = 200u;
        atlas.prevalence = prevalence;
        atlas.prevalence_count = 2u;
        atlas.communities = communities;
        atlas.community_count = 2u;
        atlas.exact_rescans = &exact;
        atlas.exact_rescan_count = 1u;

        const ex::axis_identity source = compact_axis(10u);
        const ex::axis_identity destination = compact_axis(20u);
        request.relation = {1u, 2u};
        request.structure = {3u, 4u};
        request.structure_epoch = {4u};
        request.source_axis = persistent_axis(30u);
        request.destination_axis = persistent_axis(40u);
        request.work_window = {geo::work_window_schema_version,
            geo::work_window_kind::relation_rows, {}, {5u, 6u}, destination,
            1u, 1u, &member};
        request.work_layout = {geo::work_layout_schema_version, 0u,
            request.work_window.identity, destination, 1u, &permutation,
            &permutation};
        request.relation_cover = {geo::relation_cover_schema_version, 0u,
            {7u, 1u}, request.structure_epoch, source, destination, 1u, 1u,
            0u, &component, &edge};
    }
};

void embedded_section_round_trips_and_remains_optional() {
    fixture data;
    persist::support_atlas_section_requirements_v1 required{};
    assert(persist::query_support_atlas_section_requirements_v1(
        data.atlas, &required) == persist::support_section_status_v1::success);
    assert(required.section_bytes > sizeof(geo::support_atlas_section_header_v1));
    alignas(64) std::uint8_t section[2048]{};
    persist::semantic_geometry_optional_section_v1 optional{};
    assert(persist::build_support_atlas_optional_section_v1(data.atlas,
        section, sizeof(section), &optional)
        == persist::support_section_status_v1::success);

    // Core CSG1 remains valid with no support evidence at all.
    alignas(64) std::uint8_t core_image[4096]{};
    std::uint8_t marks[1]{};
    persist::semantic_geometry_image_view_v1 core{};
    assert(persist::build_semantic_geometry_image_v1(data.request,
        {core_image, sizeof(core_image)}, {marks, sizeof(marks)}, &core)
        == persist::semantic_geometry_image_status_v1::ok);
    assert(core.section_count
        == persist::semantic_geometry_mandatory_section_count_v1);

    data.request.optional_sections = &optional;
    data.request.optional_section_count = 1u;
    alignas(64) std::uint8_t image[4096]{};
    persist::semantic_geometry_image_view_v1 built{};
    assert(persist::build_semantic_geometry_image_v1(data.request,
        {image, sizeof(image)}, {marks, sizeof(marks)}, &built)
        == persist::semantic_geometry_image_status_v1::ok);
    persist::semantic_geometry_section_view_v1 found{};
    assert(persist::find_semantic_geometry_section_v1(built,
        persist::semantic_geometry_support_atlas_section_kind_v1, &found)
        == persist::semantic_geometry_image_status_v1::ok);
    geo::support_atlas_view_v1 rebound{};
    assert(persist::rebind_support_atlas_section_v1(found, &rebound)
        == persist::support_section_status_v1::success);
    assert(rebound.evidence_identity == data.atlas.evidence_identity);
    assert(rebound.prevalence_count == 2u);
    assert(rebound.prevalence[1].weighted_destination_support == 1.5);
    assert(rebound.community_count == 2u);
    assert(rebound.communities[1].community_id == 7u);
    assert(rebound.exact_rescans[0].assigned_edge_count == 5u);

    auto corrupted = found;
    corrupted.flags ^= 1u;
    assert(persist::rebind_support_atlas_section_v1(corrupted, &rebound)
        == persist::support_section_status_v1::invalid_section);
}

void external_reference_is_pointer_free_and_storage_neutral() {
    persist::support_atlas_external_reference_v1 reference{};
    reference.evidence_identity = 10u;
    reference.relation_identity = 20u;
    reference.structure_identity = 30u;
    reference.structure_epoch = 2u;
    reference.source_axis_identity = 40u;
    reference.destination_axis_identity = 50u;
    reference.object_identity_low = 60u;
    reference.object_identity_high = 70u;
    reference.content_identity = 80u;
    reference.byte_offset = 128u;
    reference.byte_count = 1024u;
    persist::semantic_geometry_optional_section_v1 optional{};
    assert(persist::make_support_atlas_reference_section_v1(reference,
        &optional) == persist::support_section_status_v1::success);
    assert(optional.kind
        == persist::semantic_geometry_support_reference_section_kind_v1);
    assert(optional.data == &reference);
    assert(optional.data_bytes == sizeof(reference));
    reference.content_identity = 0u;
    assert(persist::make_support_atlas_reference_section_v1(reference,
        &optional) == persist::support_section_status_v1::invalid_argument);
}

} // namespace

int main() {
    embedded_section_round_trips_and_remains_optional();
    external_reference_is_pointer_free_and_storage_neutral();
    return 0;
}
