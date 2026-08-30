#include <Cellerator/compute/projection/physical_mma_hybrid.hh>
#include <Cellerator/geometry/persistence/execution_image_v2.hh>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cellerator::compute::projection {

bool validate_physical_mma_hybrid_image_v1(
    const void *data, std::size_t bytes) noexcept;

bool make_physical_mma_hybrid_cpe2_source_v1(
    const void *image,
    std::size_t image_bytes,
    std::uint64_t section_identity_low,
    std::uint64_t section_identity_high,
    std::uint64_t projection_identity_low,
    std::uint64_t projection_identity_high,
    std::uint32_t payload_section_index,
    cellpack::persistence::execution_section_source *section,
    cellpack::persistence::execution_projection_source *projection) noexcept {
    namespace persistence = cellpack::persistence;
    if (section == nullptr || projection == nullptr
        || (section_identity_low == 0u && section_identity_high == 0u)
        || (projection_identity_low == 0u && projection_identity_high == 0u)
        || image_bytes > std::numeric_limits<std::uint32_t>::max()
        || !validate_physical_mma_hybrid_image_v1(image, image_bytes))
        return false;

    persistence::execution_section_source result_section{};
    result_section.kind = persistence::execution_section_kind::projection_payload;
    result_section.schema_version = physical_mma_hybrid_schema_version_v1;
    result_section.flags = persistence::directory_device_readable;
    result_section.alignment = persistence::execution_image_v2_alignment;
    result_section.identity_low = section_identity_low;
    result_section.identity_high = section_identity_high;
    result_section.data = image;
    result_section.bytes = image_bytes;
    result_section.element_count = 1u;
    result_section.element_bytes = static_cast<std::uint32_t>(image_bytes);

    persistence::execution_projection_source result_projection{};
    result_projection.entry.identity_low = projection_identity_low;
    result_projection.entry.identity_high = projection_identity_high;
    result_projection.entry.kind =
        persistence::execution_projection_kind::architecture_specific;
    result_projection.entry.schema_version = physical_mma_hybrid_schema_version_v1;
    result_projection.entry.flags = persistence::projection_forward_capable;
    result_projection.entry.architecture_class = 70u;
    result_projection.entry.payload_section = payload_section_index;
    result_projection.entry.forward_map_section =
        persistence::invalid_directory_index;
    result_projection.entry.transpose_map_section =
        persistence::invalid_directory_index;
    result_projection.entry.scheduling_summary_section =
        persistence::invalid_directory_index;
    result_projection.entry.capability_section =
        persistence::invalid_directory_index;
    *section = result_section;
    *projection = result_projection;
    return true;
}

} // namespace cellerator::compute::projection
