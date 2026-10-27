#include "hybrid_builder_probe.hh"

#include <fstream>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <vector>

namespace projection = cellerator::compute::projection;
namespace math = cellerator::compute::math;

namespace {

bool read_words(const char* path, std::size_t count,
                std::vector<std::uint32_t>* output) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    output->resize(count);
    input.read(reinterpret_cast<char*>(output->data()),
               static_cast<std::streamsize>(count * sizeof(std::uint32_t)));
    return input.good() || (input.eof() && static_cast<std::size_t>(input.gcount())
        == count * sizeof(std::uint32_t));
}

bool write_bytes(const char* path, const std::vector<std::uint8_t>& data) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) return false;
    output.write(reinterpret_cast<const char*>(data.data()),
                 static_cast<std::streamsize>(data.size()));
    return static_cast<bool>(output);
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "usage: geometry_image_probe N E MIN_TILE_EDGES SOURCES.bin DESTINATIONS.bin IMAGE.bin\n";
        return 2;
    }
    const auto source_count = static_cast<std::uint32_t>(std::stoul(argv[1]));
    const auto edge_count = static_cast<std::size_t>(std::stoull(argv[2]));
    const auto minimum_tile_edges = static_cast<std::uint32_t>(std::stoul(argv[3]));
    std::vector<std::uint32_t> sources, destinations;
    if (!source_count || !edge_count
        || !read_words(argv[4], edge_count, &sources)
        || !read_words(argv[5], edge_count, &destinations)) {
        std::cerr << "could not read exact endpoint arrays\n";
        return 3;
    }
    projection::physical_mma_hybrid_build_request_v1 request{};
    request.structure_identity.identity_version = 1u;
    request.structure_identity.value = 0x47454f4d45545259ull;
    request.source_order.kind = math::feature_order_kind::canonical;
    request.source_order.feature_count = source_count;
    request.source_order.feature_axis_identity_version = 1u;
    request.source_order.feature_axis_identity = 0x534f555243455f31ull;
    request.destination_order.kind = math::feature_order_kind::canonical;
    request.destination_order.feature_count = source_count;
    request.destination_order.feature_axis_identity_version = 1u;
    request.destination_order.feature_axis_identity = 0x444553545f415831ull;
    request.provider_identity_low = 0x43454c4c45524154ull;
    request.provider_identity_high = 0x4f522d4e36342d31ull;
    request.capability_identity_low = 0x4e36342d574d4d41ull;
    request.capability_identity_high = 0x464c4f41542d3332ull;
    request.source_indices = sources.data();
    request.destination_indices = destinations.data();
    request.edge_count = edge_count;
    request.source_count = source_count;
    request.destination_count = source_count;
    request.dense_width = 64u;

    std::vector<std::uint8_t> image;
    const auto status = projection::build_physical_mma_hybrid_image_probe_v1(
        request, minimum_tile_edges, &image);
    if (status != projection::physical_image_build_status_v1::success
        || !projection::validate_physical_mma_hybrid_image_v1(
            image.data(), image.size())
        || !write_bytes(argv[6], image)) {
        std::cerr << "probe image construction or validation failed: "
                  << static_cast<unsigned>(status) << '\n';
        return 4;
    }
    projection::physical_mma_hybrid_header_v1 header{};
    std::memcpy(&header, image.data(), sizeof(header));
    std::cout << "{\"image_bytes\":" << image.size()
              << ",\"tile_count\":" << header.tile_count
              << ",\"mma_edge_count\":" << header.compact_slot_count
              << ",\"residual_edge_count\":"
              << header.logical_edge_count - header.compact_slot_count
              << ",\"residual_region_count\":" << header.residual_region_count
              << ",\"value_map_count\":" << header.value_map_count
              << ",\"minimum_tile_edges\":" << minimum_tile_edges << "}\n";
    return 0;
}
