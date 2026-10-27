#include "geometry_bindings.hh"

#include <Cellerator/compute/operation/sparse_support/training.hh>
#include <Cellerator/compute/projection/physical_mma_hybrid.hh>

#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
namespace projection = cellerator::compute::projection;
namespace sparse = cellerator::compute::operation::sparse_support;

namespace {

std::uint64_t hash_word(std::uint64_t hash, std::uint64_t word) noexcept {
    for (unsigned byte = 0; byte < 8; ++byte) {
        hash ^= static_cast<std::uint8_t>(word >> (byte * 8u));
        hash *= 1099511628211ull;
    }
    return hash;
}

std::uint64_t nonzero_hash(std::uint64_t value) noexcept {
    return value == 0u ? 1u : value;
}

py::dict compile_sparse_support_ce_geo(
        const std::vector<std::uint64_t>& stable_edge_ids,
        const std::vector<std::uint32_t>& sources,
        const std::vector<std::uint32_t>& destinations,
        std::uint32_t source_count, std::uint32_t destination_count,
        std::uint64_t structure_epoch, std::uint64_t projection_id) {
    if (stable_edge_ids.empty() || stable_edge_ids.size() != sources.size()
        || stable_edge_ids.size() != destinations.size() || source_count == 0u
        || destination_count == 0u || structure_epoch == 0u || projection_id == 0u
        || !std::is_sorted(stable_edge_ids.begin(), stable_edge_ids.end())
        || std::adjacent_find(stable_edge_ids.begin(), stable_edge_ids.end())
            != stable_edge_ids.end())
        throw py::value_error("support arrays must be nonempty, sorted by unique stable ID, and shape-matched");

    std::uint64_t identity = 14695981039346656037ull;
    identity = hash_word(identity, structure_epoch);
    identity = hash_word(identity, source_count);
    identity = hash_word(identity, destination_count);
    for (std::size_t edge = 0; edge < stable_edge_ids.size(); ++edge) {
        if (sources[edge] >= source_count || destinations[edge] >= destination_count)
            throw py::value_error("support endpoint ordinal is outside its logical domain");
        identity = hash_word(identity, stable_edge_ids[edge]);
        identity = hash_word(identity, sources[edge]);
        identity = hash_word(identity, destinations[edge]);
    }

    projection::physical_mma_hybrid_build_request_v1 request{};
    request.structure_identity.identity_version = 1u;
    request.structure_identity.value = nonzero_hash(identity);
    request.source_order.kind = cellerator::compute::math::feature_order_kind::canonical;
    request.source_order.feature_count = source_count;
    request.source_order.feature_axis_identity_version = 1u;
    request.source_order.feature_axis_identity = nonzero_hash(
        hash_word(14695981039346656037ull, source_count));
    request.destination_order.kind = cellerator::compute::math::feature_order_kind::canonical;
    request.destination_order.feature_count = destination_count;
    request.destination_order.feature_axis_identity_version = 1u;
    request.destination_order.feature_axis_identity = nonzero_hash(
        hash_word(7809847782465536322ull, destination_count));
    request.provider_identity_low = 0x43454c4c45524154ull; // CELLERAT
    request.provider_identity_high = 0x4f522d4e36342d31ull; // OR-N64-1
    request.capability_identity_low = 0x4e36342d574d4d41ull; // N64-WMMA
    request.capability_identity_high = 0x464c4f41542d3332ull; // FLOAT-32
    request.source_indices = sources.data();
    request.destination_indices = destinations.data();
    request.edge_count = stable_edge_ids.size();
    request.source_count = source_count;
    request.destination_count = destination_count;
    request.dense_width = 64u;

    std::vector<std::uint8_t> image;
    const auto build_status = projection::build_physical_mma_hybrid_image_v1(
        request, &image);
    if (build_status != projection::physical_image_build_status_v1::success)
        throw std::runtime_error("CE-GEO physical image builder rejected the support relation");

    std::vector<std::uint64_t> geometry_edge_ids(stable_edge_ids.size());
    std::iota(geometry_edge_ids.begin(), geometry_edge_ids.end(), std::uint64_t{0});
    std::vector<std::uint32_t> source_group_members(source_count);
    std::vector<std::uint32_t> destination_group_members(destination_count);
    std::iota(source_group_members.begin(), source_group_members.end(), 0u);
    std::iota(destination_group_members.begin(), destination_group_members.end(), 0u);
    sparse::ce_geo_n64_projection_binding_v1 binding{};
    const auto prepare_status = sparse::prepare_ce_geo_n64_projection_v1(
        image.data(), image.size(), geometry_edge_ids, sources, destinations,
        source_group_members, destination_group_members, source_count,
        destination_count, structure_epoch, projection_id, &binding);
    if (prepare_status != sparse::status_v1::success)
        throw std::runtime_error("CE-GEO image failed N64 projection preparation");

    py::dict result;
    result["image"] = py::bytes(reinterpret_cast<const char*>(image.data()), image.size());
    result["source_group_members"] = source_group_members;
    result["destination_group_members"] = destination_group_members;
    result["stable_support_edge_ids"] = stable_edge_ids;
    result["geometry_edge_ids_in_support_order"] = geometry_edge_ids;
    result["edge_value_map_indices"] = binding.edge_value_map_indices;
    result["physical_source_members"] = binding.source_members;
    result["physical_destination_members"] = binding.destination_members;
    result["logical_source_to_physical"] = binding.logical_source_to_physical;
    result["projection_id"] = projection_id;
    result["structure_epoch"] = structure_epoch;
    result["structure_identity"] = request.structure_identity.value;
    result["tile_count"] = binding.header.tile_count;
    result["residual_region_count"] = binding.header.residual_region_count;
    result["value_map_count"] = binding.header.value_map_count;
    return result;
}

} // namespace

void bind_sparse_support_geometry(py::module_& module) {
    module.def("compile_sparse_support_ce_geo", &compile_sparse_support_ce_geo,
        py::arg("edge_ids"), py::arg("sources"), py::arg("destinations"),
        py::arg("source_count"), py::arg("destination_count"),
        py::arg("structure_epoch"), py::arg("projection_id"),
        "Compile logical support into a validated canonical CE-GEO N64 image.");
}
