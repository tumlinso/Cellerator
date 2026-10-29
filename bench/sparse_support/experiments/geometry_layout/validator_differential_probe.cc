#include <Cellerator/compute/projection/physical_mma_hybrid.hh>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <vector>

namespace projection = cellerator::compute::projection;
namespace math = cellerator::compute::math;

namespace cellerator::compute::projection {
bool validate_physical_mma_hybrid_image_v1_reference(
    const void*, std::size_t) noexcept;
}

namespace {

template<typename T>
T load_record(const std::vector<unsigned char>& bytes,
              std::uint64_t offset, std::uint64_t index = 0u) {
    T value{};
    std::memcpy(&value, bytes.data() + offset + index * sizeof(T), sizeof(T));
    return value;
}

template<typename T>
void store_record(std::vector<unsigned char>* bytes, std::uint64_t offset,
                  const T& value, std::uint64_t index = 0u) {
    std::memcpy(bytes->data() + offset + index * sizeof(T), &value, sizeof(T));
}

struct generated_image {
    std::vector<unsigned char> bytes;
    projection::physical_mma_hybrid_header_v1 header{};
};

generated_image make_image(std::mt19937& rng, std::uint32_t trial) {
    constexpr std::uint32_t entity_count = 64u;
    const std::uint32_t base_count = 24u + trial % 48u;
    std::vector<std::uint32_t> sources;
    std::vector<std::uint32_t> destinations;
    std::set<std::uint64_t> pairs;
    const auto append_unique = [&](std::uint32_t source, std::uint32_t destination) {
        const std::uint64_t key = static_cast<std::uint64_t>(source) * entity_count
            + destination;
        if (pairs.insert(key).second) {
            sources.push_back(source);
            destinations.push_back(destination);
        }
    };
    append_unique(1u, 1u);
    append_unique(17u, 17u);
    while (sources.size() < base_count) {
        append_unique(static_cast<std::uint32_t>(rng() % entity_count),
                      static_cast<std::uint32_t>(rng() % entity_count));
    }
    // Duplicate endpoints become exact residual contributions in two
    // destination groups, exercising all three residual range covers.
    sources.push_back(1u);
    destinations.push_back(1u);
    sources.push_back(17u);
    destinations.push_back(17u);
    const std::uint32_t duplicate_count = trial % 3u;
    for (std::uint32_t duplicate = 0u; duplicate < duplicate_count; ++duplicate) {
        const auto selected = static_cast<std::size_t>(rng() % base_count);
        sources.push_back(sources[selected]);
        destinations.push_back(destinations[selected]);
    }

    projection::physical_mma_hybrid_build_request_v1 request{};
    request.structure_identity.identity_version = 1u;
    request.structure_identity.value = 0x1234567800000000ull + trial + 1u;
    request.source_order.kind = math::feature_order_kind::canonical;
    request.source_order.feature_count = entity_count;
    request.source_order.feature_axis_identity_version = 1u;
    request.source_order.feature_axis_identity = 0x101u;
    request.destination_order.kind = math::feature_order_kind::canonical;
    request.destination_order.feature_count = entity_count;
    request.destination_order.feature_axis_identity_version = 1u;
    request.destination_order.feature_axis_identity = 0x202u;
    request.provider_identity_low = 0x301u;
    request.capability_identity_low = 0x401u;
    request.source_indices = sources.data();
    request.destination_indices = destinations.data();
    request.edge_count = sources.size();
    request.source_count = entity_count;
    request.destination_count = entity_count;
    request.dense_width = 64u;

    generated_image result{};
    assert(projection::build_physical_mma_hybrid_image_v1(request, &result.bytes)
        == projection::physical_image_build_status_v1::success);
    result.header = load_record<projection::physical_mma_hybrid_header_v1>(
        result.bytes, 0u);
    return result;
}

bool old_accepts(const std::vector<unsigned char>& image) {
    return projection::validate_physical_mma_hybrid_image_v1_reference(
        image.data(), image.size());
}

bool new_accepts(const std::vector<unsigned char>& image) {
    return projection::validate_physical_mma_hybrid_image_v1(
        image.data(), image.size());
}

void require_same(const std::vector<unsigned char>& image, bool expected) {
    const bool old_result = old_accepts(image);
    const bool new_result = new_accepts(image);
    if (old_result != expected || new_result != expected) {
        std::cerr << "validator differential mismatch: old=" << old_result
                  << " new=" << new_result << " expected=" << expected << '\n';
        std::abort();
    }
}

bool mutate_duplicate_physical_slot(const generated_image& generated,
                                    std::vector<unsigned char>* output) {
    *output = generated.bytes;
    const auto& header = generated.header;
    for (std::uint32_t tile_index = 0u; tile_index < header.tile_count;
         ++tile_index) {
        auto tile = load_record<projection::mma_tile_v1>(generated.bytes,
            header.tile_offset, tile_index);
        if (tile.compact_slot_count < 2u) continue;
        auto first = load_record<projection::mma_compact_slot_v1>(generated.bytes,
            header.compact_slot_offset, tile.compact_slot_offset);
        auto second = load_record<projection::mma_compact_slot_v1>(generated.bytes,
            header.compact_slot_offset, tile.compact_slot_offset + 1u);
        second.row = first.row;
        second.column = first.column;
        second.dense_slot = first.dense_slot;
        store_record(output, header.compact_slot_offset, second,
                     tile.compact_slot_offset + 1u);
        auto map = load_record<projection::projection_value_map_v1>(*output,
            header.value_map_offset, second.logical_edge_index);
        map.projection_slot = first.dense_slot;
        store_record(output, header.value_map_offset, map,
                     second.logical_edge_index);
        return true;
    }
    return false;
}

} // namespace

int main() {
    constexpr std::uint32_t seed = 0x6e6f6e6du;
    constexpr std::uint32_t trials = 96u;
    std::mt19937 rng(seed);
    std::uint32_t valid_cases = 0u;
    std::uint32_t exact_malformed_cases = 0u;
    std::uint32_t ignored_padding_cases = 0u;
    std::uint32_t duplicate_slot_cases = 0u;

    for (std::uint32_t trial = 0u; trial < trials; ++trial) {
        const auto generated = make_image(rng, trial);
        require_same(generated.bytes, true);
        ++valid_cases;

        auto duplicate_id = generated.bytes;
        auto map_one = load_record<projection::projection_value_map_v1>(
            duplicate_id, generated.header.value_map_offset, 1u);
        const auto map_zero = load_record<projection::projection_value_map_v1>(
            duplicate_id, generated.header.value_map_offset, 0u);
        map_one.logical_edge_id.value = map_zero.logical_edge_id.value;
        store_record(&duplicate_id, generated.header.value_map_offset, map_one, 1u);
        require_same(duplicate_id, false);
        ++exact_malformed_cases;

        auto bad_reserved = generated.bytes;
        auto reserved_header = generated.header;
        reserved_header.reserved[0] = 1u;
        store_record(&bad_reserved, 0u, reserved_header);
        require_same(bad_reserved, false);
        ++exact_malformed_cases;

        auto bad_projection_slot = generated.bytes;
        auto bad_map = load_record<projection::projection_value_map_v1>(
            bad_projection_slot, generated.header.value_map_offset, 0u);
        bad_map.projection_slot = std::numeric_limits<std::uint32_t>::max();
        store_record(&bad_projection_slot, generated.header.value_map_offset,
                     bad_map);
        require_same(bad_projection_slot, false);
        ++exact_malformed_cases;

        auto padding = generated.bytes;
        if (generated.header.source_group_offset > generated.header.header_bytes) {
            padding[generated.header.header_bytes] ^= 0x5au;
            require_same(padding, true);
            ++ignored_padding_cases;
        }

        std::vector<unsigned char> duplicate_slot;
        if (mutate_duplicate_physical_slot(generated, &duplicate_slot)) {
            const bool old_result = old_accepts(duplicate_slot);
            const bool new_result = new_accepts(duplicate_slot);
            if (!old_result || new_result) {
                std::cerr << "duplicate physical-slot behavior mismatch: old="
                          << old_result << " new=" << new_result << '\n';
                return 2;
            }
            ++duplicate_slot_cases;
        }
    }
    if (duplicate_slot_cases == 0u || ignored_padding_cases != trials)
        return 3;
    std::cout << "validator differential probe passed seed=" << seed
              << " trials=" << trials
              << " valid=" << valid_cases
              << " exact_malformed=" << exact_malformed_cases
              << " ignored_padding=" << ignored_padding_cases
              << " old_accepts_duplicate_physical=" << duplicate_slot_cases
              << " new_rejects_duplicate_physical=" << duplicate_slot_cases
              << " reference_revision=4452dc117ad4b5240602d754506431f5f859d5f0\n";
    return 0;
}
