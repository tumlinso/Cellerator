#include <Cellerator/compute/operation/sparse_support/training.hh>
#include "../../architecture/providers/nvidia/sm70/relation_apply_n64.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <cstdint>
#include <limits>

namespace cellerator::compute::operation::sparse_support {
namespace {
constexpr std::uint32_t width = 64;
constexpr std::uint32_t block_size = 256;

std::uint32_t grid(std::uint64_t n) noexcept {
    const auto blocks = n / block_size + (n % block_size != 0);
    constexpr std::uint64_t max_grid_x = 2147483647ull;
    return blocks <= max_grid_x ? static_cast<std::uint32_t>(blocks) : 0u;
}

bool grid_covers(std::uint64_t n) noexcept {
    return n == 0 || grid(n) != 0;
}

bool mul_u64(std::uint64_t a, std::uint64_t b, std::uint64_t* result) noexcept {
    if (!result || (b && a > std::numeric_limits<std::uint64_t>::max() / b))
        return false;
    *result = a * b;
    return true;
}

__device__ std::uint32_t item_cell(const std::uint64_t* offsets,
                                   std::uint32_t cells, std::uint64_t item) {
    std::uint32_t lo = 0, hi = cells;
    while (lo + 1 < hi) {
        const std::uint32_t mid = lo + (hi - lo) / 2;
        if (item < offsets[mid]) hi = mid;
        else lo = mid;
    }
    return lo;
}

__global__ void relation_forward_kernel(
    const std::uint64_t* offsets, const std::uint64_t* ordinals,
    const std::uint64_t* gate_indices, const std::uint32_t* sources,
    const std::uint32_t* destinations, const float* probabilities,
    const std::uint8_t* admitted, std::uint32_t cells,
    std::uint64_t item_count, std::uint32_t source_count,
    std::uint32_t destination_count, const __half* x, const __half* weights,
    float* y) {
    const std::uint64_t linear = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    const std::uint64_t count = item_count * width;
    if (linear >= count) return;
    const std::uint64_t item = linear / width;
    const std::uint32_t k = static_cast<std::uint32_t>(linear % width);
    const std::uint32_t cell = item_cell(offsets, cells, item);
    const std::uint64_t edge = ordinals[item];
    const std::uint64_t gi = gate_indices[item];
    if (!admitted[gi]) return;
    const std::uint32_t source = sources[edge], destination = destinations[edge];
    if (source >= source_count || destination >= destination_count) return;
    const __half gate = __float2half_rn(probabilities[gi] * static_cast<float>(admitted[gi]));
    const __half scaled_weight = __hmul(weights[edge], gate);
    const std::uint64_t xi = (static_cast<std::uint64_t>(cell) * source_count + source)
        * width + k;
    const std::uint64_t yi = (static_cast<std::uint64_t>(cell) * destination_count
        + destination) * width + k;
    atomicAdd(y + yi, __half2float(__hmul(scaled_weight, x[xi])));
}

__global__ void relation_ce_geo_residual_kernel(
    const std::uint64_t* offsets, const std::uint64_t* ordinals,
    const std::uint64_t* gate_indices, const std::uint64_t* edge_map_indices,
    std::uint64_t value_map_count,
    const projection::projection_value_map_v1* value_maps,
    std::uint32_t residual_region_count, const std::uint64_t* projection_edge_ids,
    const std::uint32_t* sources, const std::uint32_t* destinations,
    const std::uint32_t* logical_source_to_physical,
    const float* probabilities, const std::uint8_t* admitted,
    std::uint32_t cells, std::uint64_t item_count,
    std::uint32_t logical_source_count, std::uint32_t source_count,
    std::uint32_t destination_count,
    const __half* x, const __half* weights, float* y) {
    const std::uint64_t linear = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (linear >= item_count * width) return;
    const std::uint64_t item = linear / width;
    const std::uint32_t k = static_cast<std::uint32_t>(linear % width);
    const std::uint32_t cell = item_cell(offsets, cells, item);
    const std::uint64_t edge = ordinals[item], gi = gate_indices[item];
    if (!admitted[gi]) return;
    const std::uint64_t map_index = edge_map_indices[edge];
    if (map_index >= value_map_count) return;
    const auto map = value_maps[map_index];
    if (map.region_kind != projection::physical_region_kind_v1::residual
        || map.region_index >= residual_region_count
        || map.logical_edge_id.value != projection_edge_ids[edge]) return;
    const auto source = sources[edge], destination = destinations[edge];
    if (source >= logical_source_count || destination >= destination_count) return;
    const auto physical_source = logical_source_to_physical[source];
    if (physical_source == 0xffffffffu || physical_source >= source_count) return;
    const __half gate = __float2half_rn(probabilities[gi] * static_cast<float>(admitted[gi]));
    const __half scale = __hmul(weights[edge], gate);
    const std::uint64_t xi = (static_cast<std::uint64_t>(cell) * source_count + physical_source)
        * width + k;
    const std::uint64_t yi = (static_cast<std::uint64_t>(cell) * destination_count
        + destination) * width + k;
    atomicAdd(y + yi, __half2float(__hmul(scale, x[xi])));
}

__global__ void relation_ce_geo_delta_kernel(
    const std::uint64_t* offsets, const std::uint64_t* ordinals,
    const std::uint64_t* gate_indices, const std::uint64_t* edge_map_indices,
    std::uint64_t value_map_count, const std::uint32_t* sources,
    const std::uint32_t* destinations,
    const float* probabilities, const std::uint8_t* admitted,
    std::uint64_t gate_count,
    std::uint32_t cells, std::uint64_t item_count,
    std::uint32_t logical_source_count,
    std::uint32_t destination_count, const std::uint64_t* gate_ids,
    const std::uint64_t* gate_ordinals, const std::uint64_t* edge_ids,
    const __half* x, const __half* weights, float* y) {
    const std::uint64_t linear = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (linear >= item_count * width) return;
    const std::uint64_t item = linear / width;
    const std::uint32_t k = static_cast<std::uint32_t>(linear % width);
    const std::uint32_t cell = item_cell(offsets, cells, item);
    const auto edge = ordinals[item], gate = gate_indices[item];
    if (edge_map_indices[edge] < value_map_count || gate >= gate_count
        || gate_ordinals[gate] != edge || gate_ids[gate] != edge_ids[edge]
        || !admitted[gate]) return;
    const auto logical_source = sources[edge], destination = destinations[edge];
    if (logical_source >= logical_source_count || destination >= destination_count) return;
    const __half gate_value = __float2half_rn(probabilities[gate]);
    const __half scaled_weight = __hmul(weights[edge], gate_value);
    const auto xi = (static_cast<std::uint64_t>(cell) * logical_source_count
        + logical_source) * width + k;
    const auto yi = (static_cast<std::uint64_t>(cell) * destination_count
        + destination) * width + k;
    atomicAdd(y + yi, __half2float(__hmul(scaled_weight, x[xi])));
}

__global__ void relation_dx_kernel(
    const std::uint64_t* offsets, const std::uint64_t* ordinals,
    const std::uint64_t* gate_indices, const std::uint32_t* sources,
    const std::uint32_t* destinations, const float* probabilities,
    const std::uint8_t* admitted, std::uint32_t cells,
    std::uint64_t item_count, std::uint32_t source_count,
    std::uint32_t destination_count, const __half* weights,
    const float* dy, float* dx) {
    const std::uint64_t linear = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    const std::uint64_t count = item_count * width;
    if (linear >= count) return;
    const std::uint64_t item = linear / width;
    const std::uint32_t k = static_cast<std::uint32_t>(linear % width);
    const std::uint32_t cell = item_cell(offsets, cells, item);
    const std::uint64_t edge = ordinals[item], gi = gate_indices[item];
    if (!admitted[gi]) return;
    const auto source = sources[edge], destination = destinations[edge];
    if (source >= source_count || destination >= destination_count) return;
    const __half gate = __float2half_rn(probabilities[gi] * static_cast<float>(admitted[gi]));
    const __half scale = __hmul(weights[edge], gate);
    const std::uint64_t xi = (static_cast<std::uint64_t>(cell) * source_count + source)
        * width + k;
    const std::uint64_t yi = (static_cast<std::uint64_t>(cell) * destination_count
        + destination) * width + k;
    atomicAdd(dx + xi, __half2float(scale) * dy[yi]);
}

__global__ void relation_weight_gradient_kernel(
    const std::uint64_t* offsets, const std::uint64_t* ordinals,
    const std::uint64_t* gate_indices, const std::uint32_t* sources,
    const std::uint32_t* destinations, const float* probabilities,
    const std::uint8_t* admitted, std::uint32_t cells,
    std::uint64_t item_count, std::uint32_t source_count,
    std::uint32_t destination_count, const __half* x, const float* dy,
    float* dw) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (item >= item_count) return;
    const std::uint32_t cell = item_cell(offsets, cells, item);
    const std::uint64_t edge = ordinals[item], gi = gate_indices[item];
    if (!admitted[gi]) return;
    const auto source = sources[edge], destination = destinations[edge];
    if (source >= source_count || destination >= destination_count) return;
    const float q = probabilities[gi] * static_cast<float>(admitted[gi]);
    const std::uint64_t xi = (static_cast<std::uint64_t>(cell) * source_count + source)
        * width;
    const std::uint64_t yi = (static_cast<std::uint64_t>(cell) * destination_count
        + destination) * width;
    float dot = 0.0f;
    for (std::uint32_t k = 0; k < width; ++k)
        dot += dy[yi + k] * __half2float(x[xi + k]);
    atomicAdd(dw + edge, q * dot);
}

__global__ void relation_gate_gradient_kernel(
    const std::uint64_t* offsets, const std::uint64_t* ordinals,
    const std::uint64_t* gate_indices, const std::uint32_t* sources,
    const std::uint32_t* destinations, std::uint32_t cells,
    std::uint64_t item_count, std::uint32_t source_count,
    std::uint32_t destination_count, const __half* x, const __half* weights,
    const float* dy, float* dg) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (item >= item_count) return;
    const std::uint32_t cell = item_cell(offsets, cells, item);
    const std::uint64_t edge = ordinals[item];
    const auto source = sources[edge], destination = destinations[edge];
    if (source >= source_count || destination >= destination_count) return;
    const std::uint64_t xi = (static_cast<std::uint64_t>(cell) * source_count + source)
        * width;
    const std::uint64_t yi = (static_cast<std::uint64_t>(cell) * destination_count
        + destination) * width;
    float dot = 0.0f;
    for (std::uint32_t k = 0; k < width; ++k)
        dot += dy[yi + k] * __half2float(x[xi + k]);
    dg[item] = __half2float(weights[edge]) * dot;
}

__global__ void pack_ce_geo_active_tiles_kernel(
    const std::uint64_t* item_offsets, const std::uint64_t* edge_ordinals,
    const std::uint64_t* gate_indices, const std::uint32_t* item_tile_work_indices,
    const std::uint64_t* edge_map_indices, std::uint64_t value_map_count,
    const projection::projection_value_map_v1* value_maps,
    const std::uint32_t* active_tile_ids,
    const std::uint64_t* projection_edge_ids,
    const float* probabilities, const std::uint8_t* admitted,
    const __half* weights, std::uint32_t cells, std::uint64_t item_count,
    std::uint64_t active_tile_count, __half* packed_tiles) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (item >= item_count) return;
    const std::uint32_t cell = item_cell(item_offsets, cells, item);
    const auto edge = edge_ordinals[item];
    const auto gate = gate_indices[item];
    const auto tile_work = item_tile_work_indices[item];
    const auto map_index = edge_map_indices[edge];
    if (tile_work >= active_tile_count || map_index >= value_map_count) return;
    const projection::projection_value_map_v1 map = value_maps[map_index];
    if (map.region_kind != projection::physical_region_kind_v1::mma
        || map.region_index != active_tile_ids[tile_work]
        || map.projection_slot >= 256u
        || map.logical_edge_id.value != projection_edge_ids[edge])
        return;
    const __half gate_value = __float2half_rn(
        probabilities[gate] * static_cast<float>(admitted[gate]));
    packed_tiles[tile_work * 256u + map.projection_slot] =
        __hmul(weights[edge], gate_value);
    (void)cell;
}

__global__ void scatter_physical_groups_kernel(
    const std::uint32_t* destination_members, std::uint32_t cells,
    std::uint32_t groups, const float* physical, std::uint32_t destinations,
    float* logical) {
    const std::uint64_t count = static_cast<std::uint64_t>(cells) * groups
        * 16u * width;
    const std::uint64_t linear = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (linear >= count) return;
    const std::uint32_t k = linear % width;
    const std::uint64_t row = linear / width;
    const std::uint32_t local = row % 16u;
    const std::uint64_t group_cell = row / 16u;
    const std::uint32_t group = group_cell % groups;
    const std::uint32_t cell = group_cell / groups;
    const std::uint32_t destination = destination_members[group * 16u + local];
    if (destination == 0xffffffffu
        || destination >= destinations) return;
    const std::uint64_t out = (static_cast<std::uint64_t>(cell) * destinations
        + destination) * width + k;
    logical[out] = physical[linear];
}

__global__ void gather_source_kernel(
    const std::uint32_t* physical_members, std::uint32_t physical_sources,
    std::uint32_t logical_sources, std::uint32_t cells,
    const __half* logical_input, __half* physical_input) {
    const std::uint64_t count = static_cast<std::uint64_t>(cells)
        * physical_sources * width;
    const std::uint64_t linear = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (linear >= count) return;
    const std::uint32_t k = linear % width;
    const std::uint64_t row = linear / width;
    const std::uint32_t physical = row % physical_sources;
    const std::uint32_t cell = row / physical_sources;
    const std::uint32_t logical = physical_members[physical];
    physical_input[linear] = logical == 0xffffffffu || logical >= logical_sources
        ? __float2half_rn(0.0f)
        : logical_input[(static_cast<std::uint64_t>(cell) * logical_sources
            + logical) * width + k];
}

constexpr std::uint64_t invalid_sort_key = 0xffffffffffffffffull;

__global__ void build_ce_geo_keys_kernel(
    const std::uint64_t* offsets, const std::uint64_t* ordinals,
    const std::uint64_t* gate_indices, const std::uint64_t* edge_map_indices,
    const projection::projection_value_map_v1* maps,
    const std::uint32_t* tile_destination_groups,
    const std::uint64_t* edge_ids, const std::uint64_t* projection_edge_ids,
    const std::uint64_t* gate_ids,
    const std::uint64_t* gate_ordinals, const std::uint8_t* admitted,
    std::uint32_t cells, std::uint32_t groups, std::uint32_t tile_count,
    std::uint64_t item_count, std::uint64_t map_count,
    std::uint64_t gate_count, std::uint64_t* keys) {
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (item >= item_count) return;
    keys[item] = invalid_sort_key;
    const auto edge = ordinals[item], gate = gate_indices[item];
    const auto cell = item_cell(offsets, cells, item);
    if (edge_map_indices[edge] >= map_count || gate >= gate_count
        || !admitted[gate] || gate_ordinals[gate] != edge
        || gate_ids[gate] != edge_ids[edge]) return;
    const auto map = maps[edge_map_indices[edge]];
    if (map.region_kind != projection::physical_region_kind_v1::mma
        || map.region_index >= tile_count
        || map.logical_edge_id.value != projection_edge_ids[edge])
        return;
    const auto group = tile_destination_groups[map.region_index];
    if (group >= groups) return;
    const std::uint64_t group_ordinal = static_cast<std::uint64_t>(cell) * groups + group;
    const std::uint64_t key = group_ordinal * tile_count + map.region_index;
    if (key != invalid_sort_key) keys[item] = key;
}

__global__ void mark_unique_ce_geo_keys_kernel(
    const std::uint64_t* sorted, std::uint64_t count,
    std::uint32_t* flags) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (i >= count) return;
    flags[i] = sorted[i] != invalid_sort_key
        && (i == 0 || sorted[i - 1] != sorted[i]) ? 1u : 0u;
}

__global__ void scatter_unique_ce_geo_keys_kernel(
    const std::uint64_t* sorted, const std::uint32_t* flags,
    const std::uint32_t* prefix, std::uint64_t count,
    std::uint64_t* compact) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (i < count && flags[i]) compact[prefix[i]] = sorted[i];
}

__device__ std::uint64_t key_lower_bound(const std::uint64_t* sorted,
                                         std::uint64_t count,
                                         std::uint64_t target) {
    std::uint64_t lo = 0, hi = count;
    while (lo < hi) {
        const auto mid = lo + (hi - lo) / 2;
        if (sorted[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ std::uint32_t compact_position_before(
    const std::uint64_t* sorted, const std::uint32_t* prefix,
    const std::uint32_t* flags, std::uint64_t count, std::uint64_t at) {
    if (at < count) return prefix[at];
    return count ? prefix[count - 1] + flags[count - 1] : 0u;
}

__global__ void emit_ce_geo_offsets_kernel(
    const std::uint64_t* sorted, const std::uint32_t* flags,
    const std::uint32_t* prefix, std::uint64_t item_count,
    std::uint64_t group_count, std::uint32_t tile_count,
    std::uint32_t* offsets) {
    const std::uint64_t group = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (group > group_count) return;
    const auto target = group * tile_count;
    const auto at = key_lower_bound(sorted, item_count, target);
    offsets[group] = compact_position_before(sorted, prefix, flags, item_count, at);
}

__global__ void emit_ce_geo_tiles_kernel(
    const std::uint64_t* compact, const std::uint32_t* tile_dest_groups,
    const std::uint32_t* tile_source_bases, std::uint64_t active_capacity,
    std::uint32_t groups, std::uint32_t cells, std::uint32_t sources,
    std::uint32_t tile_count, std::uint32_t* tile_ids,
    std::uint32_t* source_bases) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (i >= active_capacity) return;
    const auto key = compact[i];
    if (key == invalid_sort_key) return;
    const auto group = key / tile_count;
    const auto tile = static_cast<std::uint32_t>(key % tile_count);
    const auto cell = static_cast<std::uint32_t>(group / groups);
    if (cell >= cells || tile_dest_groups[tile] != group % groups) return;
    tile_ids[i] = tile;
    source_bases[i] = tile_source_bases[tile] + cell * sources;
}

__global__ void emit_ce_geo_item_tile_indices_kernel(
    const std::uint64_t* unsorted, const std::uint64_t* sorted,
    const std::uint32_t* prefix, std::uint64_t count,
    std::uint32_t* item_tile_indices) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (i >= count) return;
    const auto key = unsorted[i];
    if (key == invalid_sort_key) {
        item_tile_indices[i] = 0xffffffffu;
        return;
    }
    const auto at = key_lower_bound(sorted, count, key);
    item_tile_indices[i] = at < count && sorted[at] == key
        ? prefix[at] : 0xffffffffu;
}

__global__ void latent_forward_kernel(
    latent_transition_shape_v1 shape, const std::uint64_t* affected_offsets,
    const std::uint32_t* affected, std::uint64_t affected_count,
    const __half* z, const __half* u,
    const __half* a, const __half* b, const __half* bias, float* next) {
    const std::uint64_t linear = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    const std::uint64_t count = affected_count * shape.state_width;
    if (linear >= count) return;
    const std::uint32_t o = linear % shape.state_width;
    const std::uint64_t row = linear / shape.state_width;
    const std::uint32_t cell = item_cell(affected_offsets, shape.cells, row);
    const std::uint32_t mechanism = affected[row];
    float value = __half2float(bias[o]);
    for (std::uint32_t i = 0; i < shape.state_width; ++i) {
        const std::uint64_t zi = (static_cast<std::uint64_t>(cell) * shape.mechanisms
            + mechanism) * shape.state_width + i;
        value += __half2float(__hmul(
            a[static_cast<std::uint64_t>(o) * shape.state_width + i], z[zi]));
    }
    for (std::uint32_t i = 0; i < shape.input_width; ++i)
        value += __half2float(__hmul(
            b[static_cast<std::uint64_t>(o) * shape.input_width + i],
            u[static_cast<std::uint64_t>(cell) * shape.input_width + i]));
    const std::uint64_t oi = (static_cast<std::uint64_t>(cell) * shape.mechanisms
        + mechanism) * shape.state_width + o;
    next[oi] = tanhf(value);
}

__global__ void copy_f32(const float* source, float* destination, std::uint64_t count) {
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
            + threadIdx.x;
         i < count; i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
        destination[i] = source[i];
}

__global__ void copy_half_to_float(const __half* source, float* destination,
                                   std::uint64_t count) {
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
            + threadIdx.x;
         i < count; i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
        destination[i] = __half2float(source[i]);
}

__global__ void clear_affected_state_gradient(
    latent_transition_shape_v1 shape, const std::uint64_t* affected_offsets,
    const std::uint32_t* affected, std::uint64_t affected_count, float* dz) {
    const std::uint64_t linear = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    const std::uint64_t count = affected_count * shape.state_width;
    if (linear >= count) return;
    const std::uint32_t i = linear % shape.state_width;
    const std::uint64_t row = linear / shape.state_width;
    const std::uint32_t cell = item_cell(affected_offsets, shape.cells, row);
    dz[(static_cast<std::uint64_t>(cell) * shape.mechanisms + affected[row])
        * shape.state_width + i] = 0.0f;
}

__global__ void latent_vjp_kernel(
    latent_transition_shape_v1 shape, const std::uint64_t* affected_offsets,
    const std::uint32_t* affected, std::uint64_t affected_count,
    const __half* z, const __half* u,
    const __half* a, const __half* b, const float* next,
    const float* dnext, float* dz, float* du, float* da, float* db, float* dbias) {
    const std::uint64_t linear = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    const std::uint64_t count = affected_count * shape.state_width;
    if (linear >= count) return;
    const std::uint32_t o = linear % shape.state_width;
    const std::uint64_t row = linear / shape.state_width;
    const std::uint32_t cell = item_cell(affected_offsets, shape.cells, row);
    const std::uint32_t mechanism = affected[row];
    const std::uint64_t oi = (static_cast<std::uint64_t>(cell) * shape.mechanisms
        + mechanism) * shape.state_width + o;
    const float value = next[oi];
    const float g = dnext[oi] * (1.0f - value * value);
    atomicAdd(dbias + o, g);
    for (std::uint32_t i = 0; i < shape.state_width; ++i) {
        const std::uint64_t ai2 = static_cast<std::uint64_t>(o) * shape.state_width + i;
        const std::uint64_t zi = (static_cast<std::uint64_t>(cell) * shape.mechanisms
            + mechanism) * shape.state_width + i;
        atomicAdd(da + ai2, g * __half2float(z[zi]));
        atomicAdd(dz + zi, g * __half2float(a[ai2]));
    }
    for (std::uint32_t i = 0; i < shape.input_width; ++i) {
        const std::uint64_t bi = static_cast<std::uint64_t>(o) * shape.input_width + i;
        const std::uint64_t ui = static_cast<std::uint64_t>(cell) * shape.input_width + i;
        atomicAdd(db + bi, g * __half2float(u[ui]));
        atomicAdd(du + ui, g * __half2float(b[bi]));
    }
}

bool relation_valid(const edge_axis_view_v1& edges,
                    const per_cell_edge_workset_view_v1& work,
                    const edge_gate_values_view_v1& gates,
                    std::uint32_t sources, std::uint32_t destinations,
                    const void* x, const void* weights) noexcept {
    return edges.structure_epoch && work.structure_epoch == edges.structure_epoch
        && work.state_generation && work.cell_count && work.cell_offsets
        && work.item_count <= work.capacity
        && (work.item_count == 0 || (work.logical_edge_ordinals && work.gate_value_indices))
        && gates.structure_epoch == edges.structure_epoch
        && gates.state_generation == work.state_generation
        && gates.cell_count == work.cell_count && gates.candidate_count <= gates.capacity
        && (gates.candidate_count == 0 || (gates.probabilities && gates.admitted))
        && sources && destinations && x
        && (work.item_count == 0 || edges.edge_count != 0)
        && (work.item_count == 0 || gates.candidate_count != 0)
        && (work.item_count == 0 || (gates.logical_edge_ids
            && gates.logical_edge_ordinals))
        && (edges.edge_count == 0 || (weights && edges.logical_edge_ids
            && edges.source_indices && edges.destination_indices));
}

bool workset_matches(const edge_axis_view_v1& edges,
                     const per_cell_edge_workset_view_v1& work,
                     const edge_gate_values_view_v1& gates) noexcept {
    return work.structure_epoch == edges.structure_epoch
        && work.state_generation == gates.state_generation
        && work.cell_count == gates.cell_count && work.cell_offsets
        && work.item_count <= work.capacity
        && (!work.item_count || (work.logical_edge_ordinals
            && work.logical_edge_ids && work.gate_value_indices));
}

bool latent_valid(latent_transition_shape_v1 s, const std::uint64_t* affected_offsets,
                  std::uint64_t affected_count, const void* affected,
                  const void* z, const void* u,
                  const void* a, const void* b, const void* bias, const void* out) noexcept {
    std::uint64_t state_count = 0;
    const bool state_count_valid = s.cells != 0 && s.mechanisms != 0
        && s.state_width != 0
        && static_cast<std::uint64_t>(s.cells)
            <= std::numeric_limits<std::uint64_t>::max() / s.mechanisms
        && (state_count = static_cast<std::uint64_t>(s.cells) * s.mechanisms,
            state_count <= std::numeric_limits<std::uint64_t>::max() / s.state_width)
        && (state_count *= s.state_width, grid_covers(state_count));
    return s.cells && s.mechanisms && s.state_width && s.input_width
        && (affected_offsets != nullptr) && (affected_count == 0 || affected)
        && z && u && a && b && bias && out
        && state_count_valid
        && affected_count <= std::numeric_limits<std::uint64_t>::max() / s.state_width
        && grid_covers(affected_count * s.state_width);
}
} // namespace

status_v1 relation_forward_cuda_v1(
    const edge_axis_view_v1& edges, const per_cell_edge_workset_view_v1& primal,
    const edge_gate_values_view_v1& gates, const std::uint16_t* input,
    std::uint32_t sources, std::uint32_t destinations,
    const std::uint16_t* weights, float* output, void* stream_ptr) noexcept {
    std::uint64_t output_rows = 0, output_count = 0;
    if (!relation_valid(edges, primal, gates, sources, destinations, input, weights)
        || !output || !mul_u64(primal.cell_count, destinations, &output_rows)
        || !mul_u64(output_rows, width, &output_count)
        || output_count > std::numeric_limits<std::size_t>::max() / sizeof(float)
        || !grid_covers(output_count)
        || primal.item_count > std::numeric_limits<std::uint64_t>::max() / width
        || !grid_covers(primal.item_count * width))
        return status_v1::invalid_argument;
    const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    if (cudaMemsetAsync(output, 0, output_count * sizeof(float), stream) != cudaSuccess)
        return status_v1::cuda_failure;
    if (primal.item_count) {
        relation_forward_kernel<<<grid(primal.item_count * width), block_size, 0, stream>>>(
            primal.cell_offsets, primal.logical_edge_ordinals, primal.gate_value_indices,
            edges.source_indices, edges.destination_indices, gates.probabilities,
            gates.admitted, primal.cell_count, primal.item_count, sources,
            destinations, reinterpret_cast<const __half*>(input),
            reinterpret_cast<const __half*>(weights), output);
        if (cudaGetLastError() != cudaSuccess) return status_v1::cuda_failure;
    }
    return status_v1::success;
}

status_v1 relation_backward_cuda_v1(
    const edge_axis_view_v1& edges, const per_cell_edge_workset_view_v1& primal,
    const per_cell_edge_workset_view_v1& derivative,
    const edge_gate_values_view_v1& gates, const std::uint16_t* input,
    std::uint32_t sources, std::uint32_t destinations,
    const std::uint16_t* weights, const float* dy, float* dx, float* dw,
    float* dg, void* stream_ptr) noexcept {
    if (!relation_valid(edges, primal, gates, sources, destinations, input, weights)
        || derivative.structure_epoch != edges.structure_epoch
        || derivative.state_generation != primal.state_generation
        || derivative.cell_count != primal.cell_count
        || derivative.item_count > derivative.capacity
        || primal.item_count > std::numeric_limits<std::uint64_t>::max() / width
        || !grid_covers(primal.item_count * width)
        || !grid_covers(primal.item_count)
        || !grid_covers(derivative.item_count)
        || (derivative.item_count && (!derivative.cell_offsets
            || !derivative.logical_edge_ordinals || !derivative.gate_value_indices))
        || !dy || !dx || (edges.edge_count && !dw)
        || (derivative.item_count && !dg))
        return status_v1::invalid_argument;
    std::uint64_t x_rows = 0, x_count = 0, output_rows = 0, output_count = 0;
    if (!mul_u64(primal.cell_count, sources, &x_rows)
        || !mul_u64(x_rows, width, &x_count)
        || !mul_u64(primal.cell_count, destinations, &output_rows)
        || !mul_u64(output_rows, width, &output_count)
        || x_count > std::numeric_limits<std::size_t>::max() / sizeof(float)
        || output_count > std::numeric_limits<std::size_t>::max() / sizeof(float)
        || edges.edge_count > std::numeric_limits<std::size_t>::max() / sizeof(float)
        || derivative.item_count > std::numeric_limits<std::size_t>::max() / sizeof(float))
        return status_v1::insufficient_capacity;
    if (cudaMemsetAsync(dx, 0, x_count * sizeof(float), reinterpret_cast<cudaStream_t>(stream_ptr)) != cudaSuccess
        || (edges.edge_count && cudaMemsetAsync(dw, 0,
            edges.edge_count * sizeof(float), reinterpret_cast<cudaStream_t>(stream_ptr)) != cudaSuccess))
        return status_v1::cuda_failure;
    const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    if (primal.item_count) {
        relation_dx_kernel<<<grid(primal.item_count * width), block_size, 0, stream>>>(
            primal.cell_offsets, primal.logical_edge_ordinals, primal.gate_value_indices,
            edges.source_indices, edges.destination_indices, gates.probabilities,
            gates.admitted, primal.cell_count, primal.item_count, sources,
            destinations, reinterpret_cast<const __half*>(weights), dy, dx);
        relation_weight_gradient_kernel<<<grid(primal.item_count), block_size, 0, stream>>>(
            primal.cell_offsets, primal.logical_edge_ordinals, primal.gate_value_indices,
            edges.source_indices, edges.destination_indices, gates.probabilities,
            gates.admitted, primal.cell_count, primal.item_count, sources,
            destinations, reinterpret_cast<const __half*>(input), dy, dw);
        if (cudaGetLastError() != cudaSuccess) return status_v1::cuda_failure;
    }
    if (derivative.item_count) {
        relation_gate_gradient_kernel<<<grid(derivative.item_count), block_size, 0, stream>>>(
            derivative.cell_offsets, derivative.logical_edge_ordinals,
            derivative.gate_value_indices, edges.source_indices,
            edges.destination_indices, derivative.cell_count, derivative.item_count,
            sources, destinations, reinterpret_cast<const __half*>(input),
            reinterpret_cast<const __half*>(weights), dy, dg);
        if (cudaGetLastError() != cudaSuccess) return status_v1::cuda_failure;
    }
    return status_v1::success;
}

status_v1 relation_forward_ce_geo_n64_cuda_v1(
    const projection::physical_mma_hybrid_header_v1& projection_header,
    const projection::projection_value_map_v1* device_value_maps,
    std::uint64_t value_map_count,
    const std::uint64_t* device_edge_value_map_indices,
    const std::uint64_t* device_projection_edge_ids,
    const edge_axis_view_v1& edges,
    const per_cell_edge_workset_view_v1& mma_primal,
    const per_cell_edge_workset_view_v1& residual_primal,
    const edge_gate_values_view_v1& gates,
    const ce_geo_n64_workset_view_v1& physical_workset,
    const ce_geo_n64_workspace_v1& reserved,
    const std::uint16_t* input, std::uint32_t logical_sources,
    std::uint32_t physical_sources,
    std::uint32_t destinations, const std::uint16_t* weights,
    std::uint16_t* packed_tiles, float* physical_output, float* output,
    void* stream_ptr) noexcept {
    if (!device_value_maps || !device_edge_value_map_indices
        || (edges.edge_count && !device_projection_edge_ids)
        || value_map_count != projection_header.value_map_count
        || projection_header.schema_version != projection::physical_mma_hybrid_schema_version_v1
        || projection_header.dense_width != width
        || projection_header.destination_group_count != physical_workset.destination_group_count
        || physical_workset.structure_epoch != edges.structure_epoch
        || physical_workset.projection_id == 0
        || physical_workset.cell_count != mma_primal.cell_count
        || physical_workset.state_generation != mma_primal.state_generation
        || !physical_workset.cell_group_offsets
        || (physical_workset.active_tile_capacity && (!physical_workset.active_tile_ids
            || !physical_workset.active_tile_source_bases || !packed_tiles))
        || !physical_workset.destination_members
        || !physical_workset.logical_source_to_physical
        || physical_workset.active_tile_capacity > reserved.active_tile_capacity
        || (mma_primal.item_count && physical_workset.active_tile_capacity == 0)
        || mma_primal.item_count > reserved.primal_item_capacity
        || residual_primal.item_count > reserved.residual_item_capacity
        || !workset_matches(edges, mma_primal, gates)
        || !workset_matches(edges, residual_primal, gates)
        || !relation_valid(edges, mma_primal, gates, logical_sources, destinations, input, weights)
        || !relation_valid(edges, residual_primal, gates, logical_sources, destinations, input, weights)
        || (mma_primal.item_count && !physical_workset.item_tile_work_indices)
        || !physical_output || !output)
        return status_v1::invalid_argument;
    if (physical_workset.active_tile_capacity > std::numeric_limits<std::uint32_t>::max()
        || physical_sources != projection_header.source_group_count * 16u
        || static_cast<std::uint64_t>(physical_sources) * mma_primal.cell_count
            > std::numeric_limits<std::uint32_t>::max())
        return status_v1::invalid_argument;
    std::uint64_t grouped = 0, physical_count = 0, logical_count = 0;
    if (!mul_u64(mma_primal.cell_count, physical_workset.destination_group_count, &grouped)
        || !mul_u64(grouped, 16u * width, &physical_count)
        || !mul_u64(mma_primal.cell_count, destinations, &logical_count)
        || !mul_u64(logical_count, width, &logical_count)
        || physical_count > std::numeric_limits<std::size_t>::max() / sizeof(float)
        || logical_count > std::numeric_limits<std::size_t>::max() / sizeof(float)
        || physical_workset.active_tile_capacity > std::numeric_limits<std::uint64_t>::max() / (256u * sizeof(__half))
        || mma_primal.item_count > std::numeric_limits<std::uint64_t>::max() / width
        || residual_primal.item_count > std::numeric_limits<std::uint64_t>::max() / width
        || grouped > std::numeric_limits<std::uint32_t>::max()
        || physical_count > reserved.physical_output_capacity
        || !grid_covers(physical_workset.active_tile_capacity * 256u)
        || !grid_covers(mma_primal.item_count)
        || !grid_covers(residual_primal.item_count)
        || !grid_covers(residual_primal.item_count * width)
        || !grid_covers(physical_count) || !grid_covers(logical_count))
        return status_v1::insufficient_capacity;
    const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    if (cudaMemsetAsync(output, 0, logical_count * sizeof(float), stream) != cudaSuccess)
        return status_v1::cuda_failure;
    if (physical_workset.active_tile_capacity != 0) {
        const auto tile_bytes = physical_workset.active_tile_capacity * 256u * sizeof(__half);
        if (cudaMemsetAsync(packed_tiles, 0, tile_bytes, stream) != cudaSuccess)
            return status_v1::cuda_failure;
        if (cudaMemsetAsync(physical_output, 0,
                physical_count * sizeof(float), stream) != cudaSuccess)
            return status_v1::cuda_failure;
        const auto pack_count = grid(mma_primal.item_count);
        if (mma_primal.item_count) {
            pack_ce_geo_active_tiles_kernel<<<pack_count, block_size, 0, stream>>>(
                mma_primal.cell_offsets, mma_primal.logical_edge_ordinals,
                mma_primal.gate_value_indices, physical_workset.item_tile_work_indices,
                device_edge_value_map_indices, value_map_count, device_value_maps,
                physical_workset.active_tile_ids, device_projection_edge_ids,
                gates.probabilities, gates.admitted,
                reinterpret_cast<const __half*>(weights), mma_primal.cell_count,
                mma_primal.item_count, physical_workset.active_tile_capacity,
                reinterpret_cast<__half*>(packed_tiles));
            if (cudaGetLastError() != cudaSuccess) return status_v1::cuda_failure;
        }
        const auto n64_status =
            architecture::providers::nvidia::sm70::enqueue_relation_apply_n64_v1({
                reinterpret_cast<const __half*>(packed_tiles),
                static_cast<std::uint32_t>(physical_workset.active_tile_capacity),
                physical_workset.cell_group_offsets,
                static_cast<std::uint32_t>(grouped),
                physical_workset.active_tile_source_bases,
                reinterpret_cast<const __half*>(input),
                physical_sources * mma_primal.cell_count, physical_output, stream});
        if (n64_status != architecture::providers::nvidia::sm70::relation_apply_n64_status_v1::success)
            return status_v1::cuda_failure;
        scatter_physical_groups_kernel<<<grid(physical_count), block_size, 0, stream>>>(
            physical_workset.destination_members, mma_primal.cell_count,
            physical_workset.destination_group_count, physical_output, destinations, output);
        if (cudaGetLastError() != cudaSuccess) return status_v1::cuda_failure;
    }
    if (residual_primal.item_count) {
        relation_ce_geo_residual_kernel<<<grid(residual_primal.item_count * width),
            block_size, 0, stream>>>(residual_primal.cell_offsets,
            residual_primal.logical_edge_ordinals, residual_primal.gate_value_indices,
            device_edge_value_map_indices, value_map_count, device_value_maps,
            projection_header.residual_region_count, device_projection_edge_ids,
            edges.source_indices, edges.destination_indices,
            physical_workset.logical_source_to_physical, gates.probabilities,
            gates.admitted, residual_primal.cell_count, residual_primal.item_count,
            logical_sources, physical_sources, destinations,
            reinterpret_cast<const __half*>(input),
            reinterpret_cast<const __half*>(weights), output);
        if (cudaGetLastError() != cudaSuccess) return status_v1::cuda_failure;
    }
    return status_v1::success;
}

std::uint64_t ce_geo_n64_compaction_temporary_bytes_v1(
    std::uint64_t max_items) noexcept {
    if (max_items == 0) return 0;
    if (max_items > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        return 0;
    std::size_t sort_bytes = 0, scan_bytes = 0;
    const int count = static_cast<int>(max_items);
    if (cub::DeviceRadixSort::SortKeys(nullptr, sort_bytes,
            static_cast<std::uint64_t*>(nullptr), static_cast<std::uint64_t*>(nullptr),
            count, 0, 64) != cudaSuccess
        || cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes,
            static_cast<std::uint32_t*>(nullptr), static_cast<std::uint32_t*>(nullptr),
            count) != cudaSuccess)
        return 0;
    return static_cast<std::uint64_t>(sort_bytes > scan_bytes ? sort_bytes : scan_bytes);
}

status_v1 prepare_ce_geo_n64_workset_cuda_v1(
    const projection::physical_mma_hybrid_header_v1& header,
    const edge_axis_view_v1& edges,
    const per_cell_edge_workset_view_v1& admitted,
    const edge_gate_values_view_v1& gates,
    const std::uint64_t* edge_map_indices,
    const std::uint64_t* projection_edge_ids,
    const projection::projection_value_map_v1* maps,
    const std::uint32_t* tile_destination_groups,
    const std::uint32_t* tile_source_bases, std::uint32_t physical_sources,
    ce_geo_n64_workset_view_v1* output,
    ce_geo_n64_compaction_workspace_v1 workspace,
    void* stream_ptr) noexcept {
    if (!output || !edge_map_indices || (edges.edge_count && !projection_edge_ids)
        || !maps || !tile_destination_groups
        || !tile_source_bases || !output->cell_group_offsets
        || (admitted.item_count && (!admitted.logical_edge_ordinals
            || !admitted.gate_value_indices || !output->item_tile_work_indices))
        || (output->active_tile_capacity && (!output->active_tile_ids
            || !output->active_tile_source_bases))
        || !workset_matches(edges, admitted, gates)
        || header.schema_version != projection::physical_mma_hybrid_schema_version_v1
        || header.dense_width != width || header.tile_count == 0
        || header.destination_group_count == 0
        || physical_sources != header.source_group_count * 16u
        || admitted.item_count > workspace.max_items
        || output->active_tile_capacity > workspace.max_items
        || output->projection_id == 0
        || workspace.max_items > static_cast<std::uint64_t>(std::numeric_limits<int>::max())
        || (workspace.max_items && (!workspace.unsorted_keys || !workspace.sorted_keys
            || !workspace.unique_keys || !workspace.unique_flags
            || !workspace.unique_prefix || !workspace.temporary_storage))
        || (workspace.max_items && workspace.temporary_storage_bytes
            < ce_geo_n64_compaction_temporary_bytes_v1(workspace.max_items)))
        return status_v1::invalid_argument;
    std::uint64_t group_count = 0, key_domain = 0;
    if (!mul_u64(admitted.cell_count, header.destination_group_count, &group_count)
        || !mul_u64(group_count, header.tile_count, &key_domain)
        || key_domain >= invalid_sort_key || !grid_covers(group_count + 1))
        return status_v1::insufficient_capacity;
    std::uint64_t tile_domain = 0;
    if (!mul_u64(admitted.cell_count, header.tile_count, &tile_domain)
        || output->active_tile_capacity
            < (admitted.item_count < tile_domain ? admitted.item_count : tile_domain))
        return status_v1::insufficient_capacity;
    output->structure_epoch = edges.structure_epoch;
    output->state_generation = admitted.state_generation;
    output->cell_count = admitted.cell_count;
    output->destination_group_count = header.destination_group_count;
    const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    if (cudaMemsetAsync(output->cell_group_offsets, 0,
            (group_count + 1) * sizeof(std::uint32_t), stream) != cudaSuccess)
        return status_v1::cuda_failure;
    if (admitted.item_count == 0) return status_v1::success;

    if (cudaMemsetAsync(workspace.unique_keys, 0xff,
            output->active_tile_capacity * sizeof(std::uint64_t), stream) != cudaSuccess)
        return status_v1::cuda_failure;
    build_ce_geo_keys_kernel<<<grid(admitted.item_count), block_size, 0, stream>>>(
        admitted.cell_offsets, admitted.logical_edge_ordinals,
        admitted.gate_value_indices, edge_map_indices, maps,
        tile_destination_groups, edges.logical_edge_ids, projection_edge_ids,
        gates.logical_edge_ids, gates.logical_edge_ordinals, gates.admitted,
        admitted.cell_count, header.destination_group_count, header.tile_count,
        admitted.item_count, header.value_map_count, gates.candidate_count,
        workspace.unsorted_keys);
    if (cudaGetLastError() != cudaSuccess) return status_v1::cuda_failure;
    std::size_t temp_bytes = static_cast<std::size_t>(workspace.temporary_storage_bytes);
    if (cub::DeviceRadixSort::SortKeys(workspace.temporary_storage, temp_bytes,
            workspace.unsorted_keys, workspace.sorted_keys,
            static_cast<int>(admitted.item_count), 0, 64, stream) != cudaSuccess)
        return status_v1::cuda_failure;
    mark_unique_ce_geo_keys_kernel<<<grid(admitted.item_count), block_size, 0, stream>>>(
        workspace.sorted_keys, admitted.item_count, workspace.unique_flags);
    if (cudaGetLastError() != cudaSuccess) return status_v1::cuda_failure;
    temp_bytes = static_cast<std::size_t>(workspace.temporary_storage_bytes);
    if (cub::DeviceScan::ExclusiveSum(workspace.temporary_storage, temp_bytes,
            workspace.unique_flags, workspace.unique_prefix,
            static_cast<int>(admitted.item_count), stream) != cudaSuccess)
        return status_v1::cuda_failure;
    scatter_unique_ce_geo_keys_kernel<<<grid(admitted.item_count), block_size, 0, stream>>>(
        workspace.sorted_keys, workspace.unique_flags, workspace.unique_prefix,
        admitted.item_count, workspace.unique_keys);
    if (cudaGetLastError() != cudaSuccess) return status_v1::cuda_failure;
    emit_ce_geo_offsets_kernel<<<grid(group_count + 1), block_size, 0, stream>>>(
        workspace.sorted_keys, workspace.unique_flags, workspace.unique_prefix,
        admitted.item_count, group_count, header.tile_count,
        output->cell_group_offsets);
    if (cudaGetLastError() != cudaSuccess) return status_v1::cuda_failure;
    emit_ce_geo_tiles_kernel<<<grid(output->active_tile_capacity), block_size, 0, stream>>>(
        workspace.unique_keys, tile_destination_groups, tile_source_bases,
        output->active_tile_capacity, header.destination_group_count,
        admitted.cell_count, physical_sources, header.tile_count,
        output->active_tile_ids, output->active_tile_source_bases);
    if (cudaGetLastError() != cudaSuccess) return status_v1::cuda_failure;
    emit_ce_geo_item_tile_indices_kernel<<<grid(admitted.item_count), block_size, 0, stream>>>(
        workspace.unsorted_keys, workspace.sorted_keys, workspace.unique_prefix,
        admitted.item_count, output->item_tile_work_indices);
    return cudaGetLastError() == cudaSuccess
        ? status_v1::success : status_v1::cuda_failure;
}

status_v1 gather_source_to_ce_geo_n64_cuda_v1(
    const std::uint32_t* physical_members, std::uint32_t physical_sources,
    std::uint32_t logical_sources, std::uint32_t cells,
    const std::uint16_t* logical_input, std::uint16_t* physical_input,
    void* stream_ptr) noexcept {
    std::uint64_t rows = 0, count = 0;
    if (!physical_members || !physical_sources || !logical_sources || !cells
        || !logical_input || !physical_input
        || !mul_u64(cells, physical_sources, &rows)
        || !mul_u64(rows, width, &count) || !grid_covers(count))
        return status_v1::invalid_argument;
    gather_source_kernel<<<grid(count), block_size, 0,
        reinterpret_cast<cudaStream_t>(stream_ptr)>>>(physical_members,
        physical_sources, logical_sources, cells,
        reinterpret_cast<const __half*>(logical_input),
        reinterpret_cast<__half*>(physical_input));
    return cudaGetLastError() == cudaSuccess
        ? status_v1::success : status_v1::cuda_failure;
}

status_v1 relation_delta_accumulate_ce_geo_n64_cuda_v1(
    std::uint64_t value_map_count, const std::uint64_t* edge_map_indices,
    const edge_axis_view_v1& edges,
    const per_cell_edge_workset_view_v1& delta,
    const edge_gate_values_view_v1& gates,
    const std::uint16_t* logical_input, std::uint32_t logical_sources,
    std::uint32_t destinations,
    const std::uint16_t* weights, float* output, void* stream_ptr) noexcept {
    if (!edge_map_indices || !logical_input || !weights || !output
        || !relation_valid(edges, delta, gates, logical_sources, destinations,
            logical_input, weights)
        || !workset_matches(edges, delta, gates)
        || delta.item_count > std::numeric_limits<std::uint64_t>::max() / width
        || !grid_covers(delta.item_count * width))
        return status_v1::invalid_argument;
    const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    if (delta.item_count) {
        relation_ce_geo_delta_kernel<<<grid(delta.item_count * width), block_size,
            0, stream>>>(delta.cell_offsets, delta.logical_edge_ordinals,
            delta.gate_value_indices, edge_map_indices, value_map_count,
            edges.source_indices, edges.destination_indices,
            gates.probabilities,
            gates.admitted, gates.candidate_count, delta.cell_count,
            delta.item_count, logical_sources, destinations,
            gates.logical_edge_ids, gates.logical_edge_ordinals,
            edges.logical_edge_ids, reinterpret_cast<const __half*>(logical_input),
            reinterpret_cast<const __half*>(weights), output);
        if (cudaGetLastError() != cudaSuccess) return status_v1::cuda_failure;
    }
    return status_v1::success;
}

status_v1 latent_transition_cuda_v1(
    latent_transition_shape_v1 shape, const std::uint64_t* affected_offsets,
    const std::uint32_t* affected, std::uint64_t affected_count,
    const std::uint16_t* state,
    const std::uint16_t* input, const std::uint16_t* a,
    const std::uint16_t* b, const std::uint16_t* bias,
    float* next, void* stream_ptr) noexcept {
    if (!latent_valid(shape, affected_offsets, affected_count, affected,
        state, input, a, b, bias, next))
        return status_v1::invalid_argument;
    const std::uint64_t total = static_cast<std::uint64_t>(shape.cells)
        * shape.mechanisms * shape.state_width;
    const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    // Preserve the identity transition for mechanisms that were not affected.
    copy_half_to_float<<<grid(total), block_size, 0, stream>>>(
        reinterpret_cast<const __half*>(state), next, total);
    if (cudaGetLastError() != cudaSuccess) return status_v1::cuda_failure;
    const std::uint64_t count = affected_count * shape.state_width;
    if (count) {
        latent_forward_kernel<<<grid(count), block_size, 0, stream>>>(shape,
            affected_offsets, affected, affected_count, reinterpret_cast<const __half*>(state),
            reinterpret_cast<const __half*>(input), reinterpret_cast<const __half*>(a),
            reinterpret_cast<const __half*>(b), reinterpret_cast<const __half*>(bias), next);
        if (cudaGetLastError() != cudaSuccess) return status_v1::cuda_failure;
    }
    return status_v1::success;
}

status_v1 latent_transition_vjp_cuda_v1(
    latent_transition_shape_v1 shape, const std::uint64_t* affected_offsets,
    const std::uint32_t* affected, std::uint64_t affected_count,
    const std::uint16_t* state,
    const std::uint16_t* input, const std::uint16_t* a,
    const std::uint16_t* b, const float* next, const float* dnext,
    float* dz, float* du, float* da, float* db, float* dbias,
    void* stream_ptr) noexcept {
    if (!latent_valid(shape, affected_offsets, affected_count, affected,
        state, input, a, b, next, dnext)
        || !dz || !du || !da || !db || !dbias) return status_v1::invalid_argument;
    const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const std::uint64_t z_count = static_cast<std::uint64_t>(shape.cells)
        * shape.mechanisms * shape.state_width;
    const std::uint64_t u_count = static_cast<std::uint64_t>(shape.cells) * shape.input_width;
    const std::uint64_t a_count = static_cast<std::uint64_t>(shape.state_width) * shape.state_width;
    const std::uint64_t b_count = static_cast<std::uint64_t>(shape.state_width) * shape.input_width;
    if (cudaMemsetAsync(dz, 0, z_count * sizeof(float), stream) != cudaSuccess
        || cudaMemsetAsync(du, 0, u_count * sizeof(float), stream) != cudaSuccess
        || cudaMemsetAsync(da, 0, a_count * sizeof(float), stream) != cudaSuccess
        || cudaMemsetAsync(db, 0, b_count * sizeof(float), stream) != cudaSuccess
        || cudaMemsetAsync(dbias, 0, shape.state_width * sizeof(float), stream) != cudaSuccess)
        return status_v1::cuda_failure;
    const auto total = static_cast<std::uint64_t>(shape.cells) * shape.mechanisms
        * shape.state_width;
    copy_f32<<<grid(total), block_size, 0, stream>>>(dnext, dz, total);
    const auto active = affected_count * shape.state_width;
    if (active) {
        clear_affected_state_gradient<<<grid(active), block_size, 0, stream>>>(
            shape, affected_offsets, affected, affected_count, dz);
        latent_vjp_kernel<<<grid(active), block_size, 0, stream>>>(shape,
            affected_offsets, affected, affected_count, reinterpret_cast<const __half*>(state),
            reinterpret_cast<const __half*>(input), reinterpret_cast<const __half*>(a),
            reinterpret_cast<const __half*>(b), next, dnext, dz, du, da, db, dbias);
    }
    return cudaGetLastError() == cudaSuccess ? status_v1::success : status_v1::cuda_failure;
}

} // namespace cellerator::compute::operation::sparse_support
