#include <Cellerator/ops/sparse_support/training.hh>
#include <Cellerator/geometry/persistence/execution_image_v2.hh>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sparse = cellerator::compute::operation::sparse_support;
namespace projection = cellerator::compute::projection;
namespace persistence = cellpack::persistence;

namespace {
std::string current_phase = "unspecified";
std::uint64_t comparison_index = 0;

void set_phase(const char* phase) {
    current_phase = phase;
    comparison_index = 0;
}

std::size_t align64(std::size_t value) {
    return (value + 63u) & ~std::size_t{63u};
}

template<class T>
std::size_t append_records(std::vector<unsigned char>* image,
                           const T* records, std::size_t count) {
    const std::size_t offset = align64(image->size());
    image->resize(offset + sizeof(T) * count);
    std::memcpy(image->data() + offset, records, sizeof(T) * count);
    return offset;
}

std::vector<unsigned char> physical_image() {
    std::vector<unsigned char> image(sizeof(projection::physical_mma_hybrid_header_v1));
    projection::physical_group_v1 source{};
    source.group_id = 0u;
    source.semantic_component_id = 1u;
    source.member_count = 1u;
    source.padded_count = 16u;
    projection::physical_group_v1 destination = source;
    projection::mma_tile_v1 tile{};
    tile.tile_id = 0u;
    tile.source_group_index = 0u;
    tile.destination_group_index = 0u;
    tile.semantic_component_id = 1u;
    tile.occupancy_mask[0] = 1u;
    tile.compact_slot_count = 1u;
    projection::mma_compact_slot_v1 slot{};
    slot.logical_edge_index = 0u;
    projection::residual_region_v1 residual{};
    residual.region_id = 0u;
    residual.semantic_component_id = 1u;
    residual.destination_group_index = 0u;
    residual.row_count = 1u;
    residual.edge_count = 1u;
    residual.value_map_offset = 1u;
    const std::uint32_t row_offsets[] = {0u, 1u};
    const std::uint32_t columns[] = {0u};
    projection::projection_schedule_entry_v1 schedules[2]{};
    schedules[0].kind = projection::schedule_work_kind_v1::mma_tile;
    schedules[0].work_index = 0u;
    schedules[0].destination_group_index = 0u;
    schedules[0].dense_column_count = 64u;
    schedules[1].kind = projection::schedule_work_kind_v1::residual_region;
    schedules[1].work_index = 0u;
    schedules[1].destination_group_index = 0u;
    schedules[1].dense_column_count = 64u;
    projection::projection_value_map_v1 maps[2]{};
    maps[0].logical_edge_id.value = 0u;
    maps[0].region_kind = projection::physical_region_kind_v1::mma;
    maps[0].region_index = 0u;
    maps[0].projection_slot = 0u;
    maps[1].logical_edge_id.value = 1u;
    maps[1].region_kind = projection::physical_region_kind_v1::residual;
    maps[1].region_index = 0u;
    maps[1].projection_slot = 0u;
    projection::physical_mma_hybrid_header_v1 header{};
    header.structure_identity.identity_version = 1u;
    header.structure_identity.value = 0x101u;
    header.source_order.feature_count = 1u;
    header.source_order.feature_axis_identity_version = 1u;
    header.source_order.feature_axis_identity = 0x201u;
    header.destination_order.feature_count = 1u;
    header.destination_order.feature_axis_identity_version = 1u;
    header.destination_order.feature_axis_identity = 0x301u;
    header.provider_identity_low = 0x401u;
    header.capability_identity_low = 0x501u;
    header.logical_edge_count = 2u;
    header.dense_width = 64u;
    header.source_group_count = 1u;
    header.destination_group_count = 1u;
    header.tile_count = 1u;
    header.compact_slot_count = 1u;
    header.residual_region_count = 1u;
    header.schedule_entry_count = 2u;
    header.value_map_count = 2u;
    header.source_group_offset = append_records(&image, &source, 1u);
    header.destination_group_offset = append_records(&image, &destination, 1u);
    header.tile_offset = append_records(&image, &tile, 1u);
    header.compact_slot_offset = append_records(&image, &slot, 1u);
    header.residual_region_offset = append_records(&image, &residual, 1u);
    header.residual_row_offset_offset = append_records(&image, row_offsets, 2u);
    header.residual_column_index_offset = append_records(&image, columns, 1u);
    header.schedule_entry_offset = append_records(&image, schedules, 2u);
    header.value_map_offset = append_records(&image, maps, 2u);
    header.image_bytes = image.size();
    std::memcpy(image.data(), &header, sizeof(header));
    return image;
}

void check(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
void require_success(sparse::status_v1 status) {
    if (status != sparse::status_v1::success)
        throw std::runtime_error("sparse CUDA operation failed");
}

template <class T>
struct device_buffer {
    T* data = nullptr;
    std::size_t count = 0;
    explicit device_buffer(std::size_t n) : count(n) {
        if (n) check(cudaMalloc(&data, n * sizeof(T)));
    }
    ~device_buffer() { if (data) cudaFree(data); }
    device_buffer(const device_buffer&) = delete;
    device_buffer& operator=(const device_buffer&) = delete;
};

template <class T>
void upload(device_buffer<T>& dst, const std::vector<T>& src) {
    if (dst.count < src.size()) throw std::runtime_error("device buffer capacity too small");
    if (!src.empty()) check(cudaMemcpy(dst.data, src.data(),
        src.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template <class T, std::size_t N>
void upload(device_buffer<T>& dst, const T (&src)[N]) {
    if (dst.count < N) throw std::runtime_error("device buffer capacity too small");
    check(cudaMemcpy(dst.data, src, N * sizeof(T), cudaMemcpyHostToDevice));
}

__half h(float value) { return __float2half_rn(value); }
std::uint16_t half_bits(float value) { return __half_as_ushort(h(value)); }
float f(__half value) { return __half2float(value); }
float close(float actual, float expected, float tolerance = 2.0e-4f) {
    const auto index = comparison_index++;
    if (!std::isfinite(actual) || !std::isfinite(expected)
        || std::abs(actual - expected) > tolerance * (1.0f + std::abs(expected))) {
        std::ostringstream message;
        message << "CUDA sparse numeric mismatch in " << current_phase
                << " comparison " << index << ": actual=" << actual
                << " expected=" << expected << " tolerance=" << tolerance;
        throw std::runtime_error(message.str());
    }
    return actual;
}

void test_sparse_relation_and_ce_geo() {
    constexpr std::uint32_t cells = 2, logical_sources = 2, destinations = 2;
    const std::uint64_t edge_ids_host[] = {101, 205, 999};
    const std::uint32_t source_host[] = {0, 0, 1};
    const std::uint32_t destination_host[] = {0, 0, 1};
    const std::uint64_t primal_offsets_host[] = {0, 3, 5};
    const std::uint64_t primal_ordinals_host[] = {0, 1, 2, 0, 2};
    const std::uint64_t primal_ids_host[] = {101, 205, 999, 101, 999};
    const std::uint64_t primal_gate_indices_host[] = {0, 1, 2, 3, 4};
    const std::uint64_t derivative_offsets_host[] = {0, 3, 5};
    const std::uint64_t derivative_ordinals_host[] = {0, 1, 2, 0, 2};
    const std::uint64_t derivative_ids_host[] = {101, 205, 999, 101, 999};
    const std::uint64_t derivative_gate_indices_host[] = {0, 1, 2, 3, 4};
    const std::uint64_t gate_ordinals_host[] = {0, 1, 2, 0, 2};
    const std::uint64_t gate_ids_host[] = {101, 205, 999, 101, 999};
    const float probabilities_host[] = {0.5f, 0.25f, 0.7f, 0.75f, 0.4f};
    const std::uint8_t admitted_host[] = {1, 1, 0, 1, 1};
    const std::uint16_t weights_host[] = {half_bits(2.0f), half_bits(4.0f), half_bits(5.0f)};
    std::vector<std::uint16_t> logical_input_host(cells * logical_sources * 64);
    std::fill(logical_input_host.begin(), logical_input_host.begin() + 64, half_bits(1.0f));
    std::fill(logical_input_host.begin() + 64, logical_input_host.begin() + 128,
        half_bits(std::numeric_limits<float>::quiet_NaN()));
    std::fill(logical_input_host.begin() + 128, logical_input_host.begin() + 192, half_bits(3.0f));
    std::fill(logical_input_host.begin() + 192, logical_input_host.end(), half_bits(4.0f));
    std::vector<float> dy_host(cells * destinations * 64, 1.0f);

    device_buffer<std::uint64_t> edge_ids(3), primal_offsets(3), primal_ordinals(5),
        primal_ids(5), primal_gate_indices(5), derivative_offsets(3),
        derivative_ordinals(5), derivative_ids(5), derivative_gate_indices(5),
        gate_ordinals(5), gate_ids(5);
    device_buffer<std::uint32_t> sources(3), destination_indices(3);
    device_buffer<std::uint8_t> admitted(5);
    device_buffer<float> probabilities(5);
    device_buffer<std::uint16_t> weights(3), logical_input(logical_input_host.size());
    device_buffer<float> dy(dy_host.size()), output(cells * destinations * 64),
        simt_output(cells * destinations * 64),
        dx(cells * logical_sources * 64), dw(3), dg(5);
    upload(edge_ids, edge_ids_host); upload(sources, source_host);
    upload(destination_indices, destination_host);
    upload(primal_offsets, primal_offsets_host); upload(primal_ordinals, primal_ordinals_host);
    upload(primal_ids, primal_ids_host); upload(primal_gate_indices, primal_gate_indices_host);
    upload(derivative_offsets, derivative_offsets_host);
    upload(derivative_ordinals, derivative_ordinals_host); upload(derivative_ids, derivative_ids_host);
    upload(derivative_gate_indices, derivative_gate_indices_host);
    upload(gate_ordinals, gate_ordinals_host); upload(gate_ids, gate_ids_host);
    upload(probabilities, std::vector<float>(probabilities_host, probabilities_host + 5));
    upload(admitted, admitted_host); upload(weights, weights_host);
    upload(logical_input, logical_input_host); upload(dy, dy_host);

    sparse::edge_axis_view_v1 edges{7, 3, edge_ids.data, sources.data,
        destination_indices.data};
    sparse::per_cell_edge_workset_view_v1 primal{7, 9, cells,
        primal_offsets.data, primal_ordinals.data, primal_ids.data,
        primal_gate_indices.data, 5, 5};
    sparse::per_cell_edge_workset_view_v1 derivative{7, 9, cells,
        derivative_offsets.data, derivative_ordinals.data, derivative_ids.data,
        derivative_gate_indices.data, 5, 5};
    sparse::edge_gate_values_view_v1 gates{7, 9, cells, gate_ordinals.data,
        gate_ids.data, probabilities.data, admitted.data, 5, 5};

    require_success(sparse::relation_forward_cuda_v1(edges, primal, gates,
        logical_input.data, logical_sources, destinations, weights.data,
        simt_output.data, nullptr));
    std::vector<float> simt_output_host(simt_output.count);
    check(cudaMemcpy(simt_output_host.data(), simt_output.data,
        simt_output_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
    set_phase("SIMT forward");
    for (std::uint32_t k = 0; k < 64; ++k) {
        close(simt_output_host[k], 2.0f);
        close(simt_output_host[64 + k], 0.0f);
        close(simt_output_host[128 + k], 4.5f);
        close(simt_output_host[192 + k], 8.0f);
    }

    // The fixture has edge 0 in an MMA tile and edge 1 in a residual region.
    // Edge 2 is a support delta with a source/destination absent from the base.
    const auto image_host = physical_image();
    const std::uint64_t projection_ids[] = {0, 1,
        std::numeric_limits<std::uint64_t>::max()};
    const std::uint32_t source_indices[] = {0, 0, 1};
    const std::uint32_t destination_ordinals[] = {0, 0, 1};
    const std::uint32_t source_group_members[] = {0};
    const std::uint32_t destination_group_members[] = {0};
    sparse::ce_geo_n64_projection_binding_v1 binding;
    require_success(sparse::prepare_ce_geo_n64_projection_v1(image_host.data(),
        image_host.size(), projection_ids, source_indices, destination_ordinals,
        source_group_members, destination_group_members, logical_sources,
        destinations, 7, 11, &binding));
    if (binding.edge_value_map_indices[2] != std::numeric_limits<std::uint64_t>::max())
        throw std::runtime_error("delta edge unexpectedly bound to base CE-GEO projection");
    if (binding.projection_edge_ids.size() != 3
        || binding.projection_edge_ids[0] != 0
        || binding.projection_edge_ids[1] != 1
        || binding.projection_edge_ids[2] != std::numeric_limits<std::uint64_t>::max())
        throw std::runtime_error("projection edge IDs were not retained by support ordinal");
    device_buffer<std::uint8_t> image(image_host.size());
    check(cudaMemcpy(image.data, image_host.data(), image_host.size(), cudaMemcpyHostToDevice));
    device_buffer<std::uint64_t> edge_map_indices(binding.edge_value_map_indices.size());
    device_buffer<std::uint64_t> projection_edge_ids(binding.projection_edge_ids.size());
    upload(edge_map_indices, binding.edge_value_map_indices);
    upload(projection_edge_ids, binding.projection_edge_ids);
    device_buffer<std::uint32_t> tile_destinations(binding.tile_destination_groups.size()),
        tile_source_bases(binding.tile_source_bases.size()), source_members(binding.source_members.size()),
        destination_members(binding.destination_members.size()),
        logical_to_physical(binding.logical_source_to_physical.size());
    upload(tile_destinations, binding.tile_destination_groups);
    upload(tile_source_bases, binding.tile_source_bases);
    upload(source_members, binding.source_members);
    upload(destination_members, binding.destination_members);
    upload(logical_to_physical, binding.logical_source_to_physical);
    auto* device_maps = reinterpret_cast<const projection::projection_value_map_v1*>(
        reinterpret_cast<const std::byte*>(image.data) + binding.header.value_map_offset);

    const std::uint64_t max_items = primal.item_count;
    const std::uint64_t tile_capacity = 2; // min(items, cells * one stable CE-GEO tile)
    const std::uint64_t temp_bytes = sparse::ce_geo_n64_compaction_temporary_bytes_v1(max_items);
    device_buffer<std::uint64_t> unsorted_keys(max_items), sorted_keys(max_items),
        unique_keys(max_items);
    device_buffer<std::uint32_t> unique_flags(max_items), unique_prefix(max_items),
        cell_group_offsets(cells + 1), active_tile_ids(tile_capacity),
        active_tile_sources(tile_capacity), item_tile_indices(max_items);
    device_buffer<std::uint8_t> temp_storage(temp_bytes);
    device_buffer<std::uint16_t> physical_input(cells * 16 * 64),
        packed_tiles(tile_capacity * 256);
    device_buffer<float> physical_output(cells * 16 * 64),
        packed_output(cells * destinations * 64);

    sparse::ce_geo_n64_workset_view_v1 physical_workset{};
    physical_workset.structure_epoch = 7;
    physical_workset.projection_id = 11;
    physical_workset.active_tile_capacity = tile_capacity;
    physical_workset.cell_group_offsets = cell_group_offsets.data;
    physical_workset.active_tile_ids = active_tile_ids.data;
    physical_workset.active_tile_source_bases = active_tile_sources.data;
    physical_workset.item_tile_work_indices = item_tile_indices.data;
    physical_workset.destination_members = destination_members.data;
    physical_workset.logical_source_to_physical = logical_to_physical.data;
    sparse::ce_geo_n64_compaction_workspace_v1 compaction{
        max_items, unsorted_keys.data, sorted_keys.data, unique_keys.data,
        unique_flags.data, unique_prefix.data, temp_storage.data, temp_bytes};
    require_success(sparse::prepare_ce_geo_n64_workset_cuda_v1(binding.header, edges,
        primal, gates, edge_map_indices.data, projection_edge_ids.data, device_maps,
        tile_destinations.data, tile_source_bases.data, 16,
        &physical_workset, compaction, nullptr));
    require_success(sparse::gather_source_to_ce_geo_n64_cuda_v1(source_members.data,
        16, logical_sources, cells, logical_input.data, physical_input.data,
        nullptr));
    sparse::ce_geo_n64_workspace_v1 packed_capacity{tile_capacity, 5, 5,
        cells * 16 * 64, 0};
    require_success(sparse::relation_forward_ce_geo_n64_cuda_v1(binding.header,
        device_maps, binding.header.value_map_count, edge_map_indices.data,
        projection_edge_ids.data, edges, primal, primal, gates, physical_workset, packed_capacity,
        physical_input.data, logical_sources, 16, destinations, weights.data,
        packed_tiles.data, physical_output.data, packed_output.data,
        nullptr));

    // The admitted delta edge uses a newly introduced source and destination;
    // it must read logical X and accumulate after the packed base output.
    const std::uint64_t delta_offsets_host[] = {0, 0, 1};
    const std::uint64_t delta_ordinals_host[] = {2};
    const std::uint64_t delta_ids_host[] = {999};
    const std::uint64_t delta_gate_indices_host[] = {4};
    device_buffer<std::uint64_t> delta_offsets(3), delta_ordinals(1), delta_ids(1),
        delta_gate_indices(1);
    upload(delta_offsets, delta_offsets_host); upload(delta_ordinals, delta_ordinals_host);
    upload(delta_ids, delta_ids_host); upload(delta_gate_indices, delta_gate_indices_host);
    sparse::per_cell_edge_workset_view_v1 delta{7, 9, cells,
        delta_offsets.data, delta_ordinals.data, delta_ids.data,
        delta_gate_indices.data, 1, 1};
    require_success(sparse::relation_delta_accumulate_ce_geo_n64_cuda_v1(
        binding.header.value_map_count, edge_map_indices.data, edges, delta,
        gates, logical_input.data, logical_sources, destinations, weights.data,
        packed_output.data, nullptr));

    std::vector<float> output_host(packed_output.count);
    check(cudaMemcpy(output_host.data(), packed_output.data,
        output_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
    set_phase("CE-GEO packed forward and delta");
    for (std::uint32_t k = 0; k < 64; ++k) {
        close(output_host[k], 2.0f);                    // MMA edge + residual edge
        close(output_host[64 + k], 0.0f);               // no cell-0 delta
        close(output_host[128 + k], 4.5f);             // per-cell gated MMA edge
        close(output_host[192 + k], 8.0f);             // novel source/destination delta
    }

    // Independent host VJP over exactly the primal and derivative candidates.
    require_success(sparse::relation_backward_cuda_v1(edges, primal, derivative, gates,
        logical_input.data, logical_sources, destinations, weights.data,
        dy.data, dx.data, dw.data, dg.data, nullptr));
    std::vector<float> dx_host(dx.count), dw_host(dw.count), dg_host(dg.count);
    check(cudaMemcpy(dx_host.data(), dx.data, dx_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(dw_host.data(), dw.data, dw_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(dg_host.data(), dg.data, dg_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
    set_phase("relation VJP input and weight gradients");
    for (std::uint32_t k = 0; k < 64; ++k) {
        close(dx_host[k], 2.0f);
        close(dx_host[64 + k], 0.0f);
        close(dx_host[128 + k], 1.5f);
        close(dx_host[192 + k], 2.0f);
    }
    close(dw_host[0], 176.0f);
    close(dw_host[1], 16.0f);
    close(dw_host[2], 102.4f);
    close(dg_host[0], 128.0f);
    close(dg_host[1], 256.0f);
    if (!std::isnan(dg_host[2]))
        throw std::runtime_error("closed derivative candidate did not retain its gate gradient");
    close(dg_host[3], 384.0f);
    close(dg_host[4], 1280.0f);
}

void test_latent_cuda() {
    sparse::latent_transition_shape_v1 shape{2, 2, 2, 1};
    const std::uint64_t affected_offsets_host[] = {0, 1, 2};
    const std::uint32_t affected_host[] = {1, 0};
    const float state_f[] = {0.0f, 0.5f, 1.0f, -1.0f,
        2.0f, 3.0f, -2.0f, 4.0f};
    const float input_f[] = {0.25f, -0.5f};
    const float a_f[] = {0.5f, -0.25f, 0.75f, 0.25f};
    const float b_f[] = {2.0f, -1.0f};
    const float bias_f[] = {0.1f, -0.2f};
    std::vector<std::uint16_t> state_h(8), input_h(2), a_h(4), b_h(2), bias_h(2);
    std::transform(std::begin(state_f), std::end(state_f), state_h.begin(),
        [](float x) { return half_bits(x); });
    std::transform(std::begin(input_f), std::end(input_f), input_h.begin(),
        [](float x) { return half_bits(x); });
    std::transform(std::begin(a_f), std::end(a_f), a_h.begin(),
        [](float x) { return half_bits(x); });
    std::transform(std::begin(b_f), std::end(b_f), b_h.begin(),
        [](float x) { return half_bits(x); });
    std::transform(std::begin(bias_f), std::end(bias_f), bias_h.begin(),
        [](float x) { return half_bits(x); });
    device_buffer<std::uint64_t> affected_offsets(3);
    device_buffer<std::uint32_t> affected(2);
    device_buffer<std::uint16_t> state(8), input(2), a(4), b(2), bias(2);
    device_buffer<float> next(8), dnext(8), dz(8), du(2), da(4), db(2), dbias(2);
    upload(affected_offsets, affected_offsets_host); upload(affected, affected_host);
    upload(state, state_h); upload(input, input_h); upload(a, a_h);
    upload(b, b_h); upload(bias, bias_h);
    require_success(sparse::latent_transition_cuda_v1(shape, affected_offsets.data,
        affected.data, 2, state.data, input.data, a.data, b.data, bias.data,
        next.data, nullptr));
    std::vector<float> next_host(8);
    check(cudaMemcpy(next_host.data(), next.data, next_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
    set_phase("latent transition");
    std::vector<float> state_ref(8);
    for (std::size_t i = 0; i < state_ref.size(); ++i) state_ref[i] = f(h(state_f[i]));
    std::vector<float> input_ref(2), a_ref(4), b_ref(2), bias_ref(2);
    for (std::size_t i = 0; i < input_ref.size(); ++i) input_ref[i] = f(h(input_f[i]));
    for (std::size_t i = 0; i < a_ref.size(); ++i) a_ref[i] = f(h(a_f[i]));
    for (std::size_t i = 0; i < b_ref.size(); ++i) b_ref[i] = f(h(b_f[i]));
    for (std::size_t i = 0; i < bias_ref.size(); ++i) bias_ref[i] = f(h(bias_f[i]));
    std::vector<float> next_ref = state_ref;
    for (std::uint32_t cell = 0; cell < 2; ++cell) {
        const std::uint32_t mechanism = affected_host[cell];
        for (std::uint32_t o = 0; o < 2; ++o) {
            float value = bias_ref[o];
            for (std::uint32_t i = 0; i < 2; ++i) {
                const auto zi = (cell * 2 + mechanism) * 2 + i;
                value += f(__hmul(h(a_ref[o * 2 + i]), h(state_ref[zi])));
            }
            value += f(__hmul(h(b_ref[o]), h(input_ref[cell])));
            next_ref[(cell * 2 + mechanism) * 2 + o] = std::tanh(value);
        }
    }
    for (std::size_t i = 0; i < 8; ++i) close(next_host[i], next_ref[i]);

    std::vector<float> dnext_host(8, 1.0f);
    upload(dnext, dnext_host);
    require_success(sparse::latent_transition_vjp_cuda_v1(shape, affected_offsets.data,
        affected.data, 2, state.data, input.data, a.data, b.data, next.data,
        dnext.data, dz.data, du.data, da.data, db.data, dbias.data,
        nullptr));
    std::vector<float> dz_host(8), du_host(2), da_host(4), db_host(2), dbias_host(2);
    check(cudaMemcpy(dz_host.data(), dz.data, dz_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(du_host.data(), du.data, du_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(da_host.data(), da.data, da_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(db_host.data(), db.data, db_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(dbias_host.data(), dbias.data, dbias_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
    set_phase("latent transition VJP");
    std::vector<float> dz_ref(8, 1.0f), du_ref(2), da_ref(4), db_ref(2), dbias_ref(2);
    for (std::uint32_t cell = 0; cell < 2; ++cell) {
        const std::uint32_t mechanism = affected_host[cell];
        for (std::uint32_t i = 0; i < 2; ++i)
            dz_ref[(cell * 2 + mechanism) * 2 + i] = 0.0f;
    }
    for (std::uint32_t cell = 0; cell < 2; ++cell) {
        const std::uint32_t mechanism = affected_host[cell];
        float dpre[2]{};
        for (std::uint32_t o = 0; o < 2; ++o) {
            const auto oi = (cell * 2 + mechanism) * 2 + o;
            dpre[o] = 1.0f - next_ref[oi] * next_ref[oi];
            dbias_ref[o] += dpre[o];
            for (std::uint32_t i = 0; i < 2; ++i) {
                const auto ai = o * 2 + i;
                const auto zi = (cell * 2 + mechanism) * 2 + i;
                da_ref[ai] += dpre[o] * state_ref[zi];
                dz_ref[zi] += dpre[o] * a_ref[ai];
            }
            db_ref[o] += dpre[o] * input_ref[cell];
            du_ref[cell] += dpre[o] * b_ref[o];
        }
    }
    for (std::size_t i = 0; i < dz_host.size(); ++i) close(dz_host[i], dz_ref[i]);
    for (std::size_t i = 0; i < du_host.size(); ++i) close(du_host[i], du_ref[i]);
    for (std::size_t i = 0; i < da_host.size(); ++i) close(da_host[i], da_ref[i]);
    for (std::size_t i = 0; i < db_host.size(); ++i) close(db_host[i], db_ref[i]);
    for (std::size_t i = 0; i < dbias_host.size(); ++i) close(dbias_host[i], dbias_ref[i]);
}
} // namespace

int main() {
    int device_count = 0;
    check(cudaGetDeviceCount(&device_count));
    if (device_count == 0) throw std::runtime_error("CUDA device required for sparse numeric test");
    test_sparse_relation_and_ce_geo();
    test_latent_cuda();
    return 0;
}
