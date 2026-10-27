#include <Cellerator/compute/operation/sparse_support/training.hh>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace cellerator::compute::operation::sparse_support {
namespace {
bool product(std::uint64_t a, std::uint64_t b, std::uint64_t* out) noexcept {
    if (out == nullptr || (b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b))
        return false;
    *out = a * b;
    return true;
}

bool valid_edges(const edge_axis_view_v1& e) noexcept {
    if (e.structure_epoch == 0 || (e.edge_count != 0 && (!e.logical_edge_ids
        || !e.source_indices || !e.destination_indices))) return false;
    for (std::uint64_t i = 1; i < e.edge_count; ++i)
        if (e.logical_edge_ids[i - 1] >= e.logical_edge_ids[i]) return false;
    return true;
}

bool valid_workset(const edge_axis_view_v1& e,
                   const per_cell_edge_workset_view_v1& w,
                   bool require_gate_indices) noexcept {
    if (w.structure_epoch != e.structure_epoch || w.state_generation == 0
        || w.cell_count == 0 || !w.cell_offsets
        || (w.item_count != 0 && (!w.logical_edge_ordinals || !w.logical_edge_ids
            || (require_gate_indices && !w.gate_value_indices)))
        || w.item_count > w.capacity) return false;
    if (w.cell_offsets[0] != 0 || w.cell_offsets[w.cell_count] != w.item_count)
        return false;
    for (std::uint32_t c = 0; c < w.cell_count; ++c) {
        if (w.cell_offsets[c] > w.cell_offsets[c + 1]
            || w.cell_offsets[c + 1] > w.item_count) return false;
        std::uint64_t prior = 0;
        bool first = true;
        for (std::uint64_t j = w.cell_offsets[c]; j < w.cell_offsets[c + 1]; ++j) {
            const auto ordinal = w.logical_edge_ordinals[j];
            if (ordinal >= e.edge_count || e.logical_edge_ids[ordinal] != w.logical_edge_ids[j]
                || (!first && ordinal <= prior)) return false;
            prior = ordinal;
            first = false;
        }
    }
    return true;
}

bool valid_gates(const edge_gate_values_view_v1& g, std::uint64_t epoch,
                 std::uint64_t generation, std::uint32_t cells) noexcept {
    if (g.structure_epoch != epoch || g.state_generation != generation
        || g.cell_count != cells || g.candidate_count > g.capacity
        || (g.candidate_count && (!g.logical_edge_ordinals || !g.logical_edge_ids
            || !g.probabilities || !g.admitted))) return false;
    for (std::uint64_t i = 0; i < g.candidate_count; ++i)
        if (!std::isfinite(g.probabilities[i]) || g.probabilities[i] < 0.0f
            || g.probabilities[i] > 1.0f || g.admitted[i] > 1u) return false;
    return true;
}

bool valid_endpoints(const edge_axis_view_v1& e, std::uint32_t sources,
                     std::uint32_t destinations) noexcept {
    for (std::uint64_t i = 0; i < e.edge_count; ++i)
        if (e.source_indices[i] >= sources || e.destination_indices[i] >= destinations)
            return false;
    return true;
}

bool workset_gate_indices_valid(const per_cell_edge_workset_view_v1& w,
                                const edge_axis_view_v1& e,
                                const edge_gate_values_view_v1& g) noexcept {
    for (std::uint64_t i = 0; i < w.item_count; ++i) {
        const auto gi = w.gate_value_indices[i], edge = w.logical_edge_ordinals[i];
        if (gi >= g.candidate_count || g.logical_edge_ordinals[gi] != edge
            || g.logical_edge_ids[gi] != e.logical_edge_ids[edge]) return false;
    }
    return true;
}

bool valid_shape(latent_transition_shape_v1 s,
                 std::span<const std::uint64_t> affected_offsets,
                 std::span<const std::uint32_t> affected,
                 std::span<const float> state, std::span<const float> input,
                 std::span<const float> a, std::span<const float> b,
                 std::span<const float> bias, std::span<const float> output) noexcept {
    std::uint64_t state_count, input_count, aa, bb;
    if (!s.cells || !s.mechanisms || !s.state_width || !s.input_width
        || !product(s.cells, s.mechanisms, &state_count)
        || !product(state_count, s.state_width, &state_count)
        || !product(s.cells, s.input_width, &input_count)
        || !product(s.state_width, s.state_width, &aa)
        || !product(s.state_width, s.input_width, &bb)
        || state.size() != state_count || input.size() != input_count
        || a.size() != aa || b.size() != bb || bias.size() != s.state_width
        || output.size() != state_count) return false;
    if (affected_offsets.size() != static_cast<std::size_t>(s.cells) + 1
        || affected_offsets.front() != 0 || affected_offsets.back() != affected.size())
        return false;
    for (std::uint32_t c = 0; c < s.cells; ++c) {
        if (affected_offsets[c] > affected_offsets[c + 1]
            || affected_offsets[c + 1] > affected.size()) return false;
        for (auto i = affected_offsets[c]; i < affected_offsets[c + 1]; ++i)
            if (affected[i] >= s.mechanisms
                || (i > affected_offsets[c] && affected[i - 1] >= affected[i])) return false;
    }
    return true;
}

template <class T>
bool image_record(const std::byte* image, std::size_t bytes,
                  std::uint64_t offset, std::uint64_t index, T* out) noexcept {
    if (!image || !out || offset > bytes
        || index > (std::numeric_limits<std::uint64_t>::max() - offset) / sizeof(T))
        return false;
    const std::uint64_t begin = offset + index * sizeof(T);
    if (begin > bytes || sizeof(T) > bytes - begin) return false;
    std::memcpy(out, image + begin, sizeof(T));
    return true;
}
} // namespace

sparse_parameter_owner_v1::sparse_parameter_owner_v1(
    indexed::indexed_axis axis, std::vector<indexed::identity> ids,
    std::span<const float> initial, std::vector<parameter_block_v1> blocks,
    int device, std::uint32_t reader_capacity)
    : owner_(std::make_shared<indexed::mechanism_parameter_owner>(axis,
          std::move(ids), initial, device, reader_capacity)), blocks_(std::move(blocks)) {
    std::uint64_t cursor = 0;
    for (std::size_t block_index = 0; block_index < blocks_.size(); ++block_index) {
        const auto& block = blocks_[block_index];
        std::uint64_t count = 1;
        for (const auto extent : block.shape) {
            if (extent && count > std::numeric_limits<std::uint64_t>::max() / extent)
                throw std::invalid_argument("sparse parameter block shape overflows");
            count *= extent;
        }
        if (block.name.empty() || block.offset != cursor || block.numel != count
            || block.numel > initial.size() - std::min<std::uint64_t>(cursor, initial.size()))
            throw std::invalid_argument("sparse parameter blocks must form a named contiguous partition");
        for (std::size_t prior = 0; prior < block_index; ++prior)
            if (blocks_[prior].name == block.name)
                throw std::invalid_argument("sparse parameter block names must be unique");
        cursor += block.numel;
    }
    if (cursor != initial.size())
        throw std::invalid_argument("sparse parameter blocks do not cover the parameter vector");
}

const std::shared_ptr<indexed::mechanism_parameter_owner>&
sparse_parameter_owner_v1::native_owner() const noexcept { return owner_; }
const std::vector<parameter_block_v1>&
sparse_parameter_owner_v1::blocks() const noexcept { return blocks_; }
std::uint64_t sparse_parameter_owner_v1::parameter_count() const noexcept {
    return owner_->size();
}

status_v1 prepare_ce_geo_n64_projection_v1(
    const void* data, std::size_t bytes,
    std::span<const std::uint64_t> projection_edge_ids,
    std::span<const std::uint32_t> support_sources,
    std::span<const std::uint32_t> support_destinations,
    std::span<const std::uint32_t> source_members,
    std::span<const std::uint32_t> destination_members,
    std::uint32_t logical_source_count, std::uint32_t logical_destination_count,
    std::uint64_t structure_epoch, std::uint64_t projection_id,
    ce_geo_n64_projection_binding_v1* output) noexcept {
    if (!data || !output || !structure_epoch || !projection_id
        || support_sources.size() != projection_edge_ids.size()
        || support_destinations.size() != projection_edge_ids.size()
        || !projection::validate_physical_mma_hybrid_image_v1(data, bytes))
        return status_v1::invalid_projection;
    projection::physical_mma_hybrid_header_v1 header{};
    if (bytes < sizeof(header)) return status_v1::invalid_projection;
    std::memcpy(&header, data, sizeof(header));
    if (header.dense_width != 64 || logical_source_count == 0
        || logical_destination_count == 0)
        return status_v1::invalid_projection;
    for (std::size_t i = 0; i < projection_edge_ids.size(); ++i)
        if (support_sources[i] >= logical_source_count
            || support_destinations[i] >= logical_destination_count)
            return status_v1::invalid_argument;
    const auto* image = static_cast<const std::byte*>(data);
    try {
        ce_geo_n64_projection_binding_v1 result;
        result.structure_epoch = structure_epoch;
        result.projection_id = projection_id;
        result.header = header;
        result.projection_edge_ids.assign(projection_edge_ids.begin(), projection_edge_ids.end());
        result.source_members.assign(static_cast<std::size_t>(header.source_group_count) * 16,
            std::numeric_limits<std::uint32_t>::max());
        result.destination_members.assign(
            static_cast<std::size_t>(header.destination_group_count) * 16,
            std::numeric_limits<std::uint32_t>::max());
        result.logical_source_to_physical.assign(logical_source_count,
            std::numeric_limits<std::uint32_t>::max());
        std::vector<std::uint8_t> destination_seen(logical_destination_count, 0u);
        result.edge_value_map_indices.assign(projection_edge_ids.size(),
            std::numeric_limits<std::uint64_t>::max());
        result.tile_destination_groups.resize(header.tile_count);
        result.tile_source_bases.resize(header.tile_count);

        std::vector<std::pair<std::uint64_t, std::uint64_t>> maps;
        maps.reserve(header.value_map_count);
        for (std::uint64_t i = 0; i < header.value_map_count; ++i) {
            projection::projection_value_map_v1 map{};
            if (!image_record(image, bytes, header.value_map_offset, i, &map))
                return status_v1::invalid_projection;
            maps.emplace_back(map.logical_edge_id.value, i);
        }
        std::sort(maps.begin(), maps.end());
        for (std::size_t i = 1; i < maps.size(); ++i)
            if (maps[i - 1].first == maps[i].first) return status_v1::invalid_projection;
        for (std::size_t i = 0; i < projection_edge_ids.size(); ++i) {
            if (projection_edge_ids[i] == std::numeric_limits<std::uint64_t>::max())
                continue;
            const auto found = std::lower_bound(maps.begin(), maps.end(),
                std::pair<std::uint64_t, std::uint64_t>{projection_edge_ids[i], 0});
            if (found != maps.end() && found->first == projection_edge_ids[i])
                result.edge_value_map_indices[i] = found->second;
        }

        for (std::uint32_t g = 0; g < header.source_group_count; ++g) {
            projection::physical_group_v1 group{};
            if (!image_record(image, bytes, header.source_group_offset, g, &group)
                || group.member_count > 16 || group.padded_count != 16
                || group.group_id != g || group.member_offset > source_members.size()
                || group.member_count > source_members.size() - group.member_offset)
                return status_v1::invalid_projection;
            const std::uint64_t base = static_cast<std::uint64_t>(g) * 16;
            for (std::uint32_t i = 0; i < group.member_count; ++i) {
                result.source_members[base + i] = source_members[group.member_offset + i];
                const auto logical = source_members[group.member_offset + i];
                if (logical >= logical_source_count
                    || result.logical_source_to_physical[logical]
                        != std::numeric_limits<std::uint32_t>::max())
                    return status_v1::invalid_projection;
                result.logical_source_to_physical[logical] =
                    static_cast<std::uint32_t>(base + i);
            }
            for (std::uint32_t i = group.member_count; i < 16; ++i)
                result.source_members[base + i] = std::numeric_limits<std::uint32_t>::max();
        }
        for (std::uint32_t g = 0; g < header.destination_group_count; ++g) {
            projection::physical_group_v1 group{};
            if (!image_record(image, bytes, header.destination_group_offset, g, &group)
                || group.member_count > 16 || group.padded_count != 16
                || group.group_id != g || group.member_offset > destination_members.size()
                || group.member_count > destination_members.size() - group.member_offset)
                return status_v1::invalid_projection;
            const std::uint64_t base = static_cast<std::uint64_t>(g) * 16;
            for (std::uint32_t i = 0; i < group.member_count; ++i) {
                result.destination_members[base + i] = destination_members[group.member_offset + i];
                const auto logical = destination_members[group.member_offset + i];
                if (logical >= logical_destination_count || destination_seen[logical])
                    return status_v1::invalid_projection;
                destination_seen[logical] = 1u;
            }
            for (std::uint32_t i = group.member_count; i < 16; ++i)
                result.destination_members[base + i] = std::numeric_limits<std::uint32_t>::max();
        }
        for (std::uint32_t t = 0; t < header.tile_count; ++t) {
            projection::mma_tile_v1 tile{};
            if (!image_record(image, bytes, header.tile_offset, t, &tile)
                || tile.source_group_index >= header.source_group_count
                || tile.destination_group_index >= header.destination_group_count)
                return status_v1::invalid_projection;
            result.tile_destination_groups[t] = tile.destination_group_index;
            result.tile_source_bases[t] = tile.source_group_index * 16u;
        }
        for (std::size_t edge = 0; edge < projection_edge_ids.size(); ++edge) {
            const auto map_index = result.edge_value_map_indices[edge];
            if (map_index == std::numeric_limits<std::uint64_t>::max()) continue;
            projection::projection_value_map_v1 map{};
            if (!image_record(image, bytes, header.value_map_offset, map_index, &map)
                || map.logical_edge_id.value != projection_edge_ids[edge])
                return status_v1::invalid_projection;
            if (map.region_kind == projection::physical_region_kind_v1::mma) {
                projection::mma_tile_v1 tile{};
                if (map.region_index >= header.tile_count
                    || !image_record(image, bytes, header.tile_offset, map.region_index, &tile))
                    return status_v1::invalid_projection;
                bool matched = false;
                for (std::uint32_t local = 0; local < tile.compact_slot_count; ++local) {
                    projection::mma_compact_slot_v1 slot{};
                    if (!image_record(image, bytes, header.compact_slot_offset,
                            tile.compact_slot_offset + local, &slot))
                        return status_v1::invalid_projection;
                    if (slot.logical_edge_index != map_index) continue;
                    const auto source = result.source_members[
                        static_cast<std::uint64_t>(tile.source_group_index) * 16 + slot.column];
                    const auto destination = result.destination_members[
                        static_cast<std::uint64_t>(tile.destination_group_index) * 16 + slot.row];
                    if (slot.dense_slot != map.projection_slot
                        || source != support_sources[edge]
                        || destination != support_destinations[edge])
                        return status_v1::invalid_projection;
                    matched = true;
                    break;
                }
                if (!matched) return status_v1::invalid_projection;
            } else {
                projection::residual_region_v1 region{};
                if (map.region_index >= header.residual_region_count
                    || !image_record(image, bytes, header.residual_region_offset,
                        map.region_index, &region)
                    || map.projection_slot >= region.edge_count)
                    return status_v1::invalid_projection;
                const auto row_offset_base = header.residual_row_offset_offset
                    + static_cast<std::uint64_t>(region.row_offset_index) * sizeof(std::uint32_t);
                std::uint32_t target_row = std::numeric_limits<std::uint32_t>::max();
                for (std::uint32_t row = 0; row < region.row_count; ++row) {
                    std::uint32_t begin = 0, end = 0;
                    if (!image_record(image, bytes, row_offset_base, row, &begin)
                        || !image_record(image, bytes, row_offset_base, row + 1, &end))
                        return status_v1::invalid_projection;
                    if (map.projection_slot >= begin && map.projection_slot < end) {
                        target_row = row;
                        break;
                    }
                }
                std::uint32_t source_physical = 0;
                const auto column_offset = header.residual_column_index_offset
                    + static_cast<std::uint64_t>(region.column_index_offset
                        + map.projection_slot) * sizeof(std::uint32_t);
                if (target_row == std::numeric_limits<std::uint32_t>::max()
                    || !image_record(image, bytes, column_offset, 0, &source_physical)
                    || source_physical >= result.source_members.size()
                    || result.source_members[source_physical] != support_sources[edge]
                    || result.destination_members[
                        static_cast<std::uint64_t>(region.destination_group_index) * 16
                            + target_row] != support_destinations[edge])
                    return status_v1::invalid_projection;
            }
        }
        *output = std::move(result);
        return status_v1::success;
    } catch (...) {
        return status_v1::insufficient_capacity;
    }
}

status_v1 forward_f32_reference_v1(
    const edge_axis_view_v1& e, std::uint32_t source_count,
    std::uint32_t destination_count, std::uint32_t cells,
    std::span<const float> x, std::span<const float> w,
    std::span<const float> q, std::span<float> y) noexcept {
    std::uint64_t xn, yn, qn;
    if (!valid_edges(e) || !source_count || !destination_count || !cells
        || !product(cells, source_count, &xn) || !product(xn, 64, &xn)
        || !product(cells, destination_count, &yn) || !product(yn, 64, &yn)
        || !product(cells, e.edge_count, &qn) || x.size() != xn || w.size() != e.edge_count
        || q.size() != qn || y.size() != yn || !valid_endpoints(e, source_count, destination_count))
        return status_v1::invalid_argument;
    std::fill(y.begin(), y.end(), 0.0f);
    for (std::uint32_t c = 0; c < cells; ++c)
        for (std::uint64_t edge = 0; edge < e.edge_count; ++edge) {
            const float gate = q[static_cast<std::uint64_t>(c) * e.edge_count + edge];
            if (gate == 0.0f) continue;
            const auto s = e.source_indices[edge], d = e.destination_indices[edge];
            const float scale = w[edge] * gate;
            const auto xi = (static_cast<std::uint64_t>(c) * source_count + s) * 64;
            const auto yi = (static_cast<std::uint64_t>(c) * destination_count + d) * 64;
            for (std::uint32_t k = 0; k < 64; ++k) y[yi + k] += scale * x[xi + k];
        }
    return status_v1::success;
}

status_v1 backward_f32_reference_v1(
    const edge_axis_view_v1& e, std::uint32_t source_count,
    std::uint32_t destination_count, std::uint32_t cells,
    std::span<const float> x, std::span<const float> w,
    std::span<const float> q, std::span<const float> dy,
    std::span<float> dx, std::span<float> dw, std::span<float> dq) noexcept {
    std::uint64_t xn, yn, qn;
    if (!valid_edges(e) || !source_count || !destination_count || !cells
        || !product(cells, source_count, &xn) || !product(xn, 64, &xn)
        || !product(cells, destination_count, &yn) || !product(yn, 64, &yn)
        || !product(cells, e.edge_count, &qn) || x.size() != xn || w.size() != e.edge_count
        || q.size() != qn || dy.size() != yn || dx.size() != xn
        || dw.size() != e.edge_count || dq.size() != qn)
        return status_v1::invalid_argument;
    if (!valid_endpoints(e, source_count, destination_count)) return status_v1::invalid_argument;
    std::fill(dx.begin(), dx.end(), 0.0f);
    std::fill(dw.begin(), dw.end(), 0.0f);
    std::fill(dq.begin(), dq.end(), 0.0f);
    for (std::uint32_t c = 0; c < cells; ++c)
        for (std::uint64_t edge = 0; edge < e.edge_count; ++edge) {
            const auto s = e.source_indices[edge], d = e.destination_indices[edge];
            const auto gi = static_cast<std::uint64_t>(c) * e.edge_count + edge;
            const bool admitted = q[gi] != 0.0f;
            const auto xi = (static_cast<std::uint64_t>(c) * source_count + s) * 64;
            const auto yi = (static_cast<std::uint64_t>(c) * destination_count + d) * 64;
            float dot = 0;
            for (std::uint32_t k = 0; k < 64; ++k) dot += dy[yi + k] * x[xi + k];
            dq[gi] = w[edge] * dot;
            if (admitted) {
                dw[edge] += q[gi] * dot;
                const float scale = w[edge] * q[gi];
                for (std::uint32_t k = 0; k < 64; ++k) dx[xi + k] += scale * dy[yi + k];
            }
        }
    return status_v1::success;
}

status_v1 sparse_forward_f32_reference_v1(
    const edge_axis_view_v1& e, const per_cell_edge_workset_view_v1& primal,
    const edge_gate_values_view_v1& gates, std::uint32_t source_count,
    std::uint32_t destination_count, std::span<const float> x,
    std::span<const float> weights, std::span<float> y) noexcept {
    std::uint64_t xn, yn;
    if (!valid_edges(e) || !valid_workset(e, primal, true)
        || !valid_gates(gates, e.structure_epoch, primal.state_generation, primal.cell_count)
        || !source_count || !destination_count
        || !product(primal.cell_count, source_count, &xn) || !product(xn, 64, &xn)
        || !product(primal.cell_count, destination_count, &yn) || !product(yn, 64, &yn)
        || x.size() != xn || weights.size() != e.edge_count || y.size() != yn)
        return status_v1::invalid_argument;
    if (!valid_endpoints(e, source_count, destination_count)
        || !workset_gate_indices_valid(primal, e, gates)) return status_v1::invalid_argument;
    std::fill(y.begin(), y.end(), 0.0f);
    for (std::uint32_t c = 0; c < primal.cell_count; ++c)
        for (auto i = primal.cell_offsets[c]; i < primal.cell_offsets[c + 1]; ++i) {
            const auto edge = primal.logical_edge_ordinals[i];
            const auto gi = primal.gate_value_indices[i];
            if (!gates.admitted[gi]) continue;
            const auto s = e.source_indices[edge], d = e.destination_indices[edge];
            const float q = gates.probabilities[gi];
            const auto xi = (static_cast<std::uint64_t>(c) * source_count + s) * 64;
            const auto yi = (static_cast<std::uint64_t>(c) * destination_count + d) * 64;
            for (std::uint32_t k = 0; k < 64; ++k) y[yi + k] += weights[edge] * q * x[xi + k];
        }
    return status_v1::success;
}

status_v1 sparse_backward_f32_reference_v1(
    const edge_axis_view_v1& e, const per_cell_edge_workset_view_v1& primal,
    const per_cell_edge_workset_view_v1& derivative,
    const edge_gate_values_view_v1& gates, std::uint32_t source_count,
    std::uint32_t destination_count, std::span<const float> x,
    std::span<const float> weights, std::span<const float> dy,
    std::span<float> dx, std::span<float> dw, std::span<float> dg) noexcept {
    std::uint64_t xn, yn;
    if (!valid_edges(e) || !valid_workset(e, primal, true)
        || !valid_workset(e, derivative, true)
        || primal.cell_count != derivative.cell_count
        || primal.state_generation != derivative.state_generation
        || !valid_gates(gates, e.structure_epoch, primal.state_generation, primal.cell_count)
        || !source_count || !destination_count
        || !product(primal.cell_count, source_count, &xn) || !product(xn, 64, &xn)
        || !product(primal.cell_count, destination_count, &yn) || !product(yn, 64, &yn)
        || x.size() != xn || weights.size() != e.edge_count || dy.size() != yn
        || dx.size() != xn || dw.size() != e.edge_count || dg.size() != derivative.item_count)
        return status_v1::invalid_argument;
    if (!valid_endpoints(e, source_count, destination_count)
        || !workset_gate_indices_valid(primal, e, gates)
        || !workset_gate_indices_valid(derivative, e, gates)) return status_v1::invalid_argument;
    std::fill(dx.begin(), dx.end(), 0.0f);
    std::fill(dw.begin(), dw.end(), 0.0f);
    std::fill(dg.begin(), dg.end(), 0.0f);
    for (std::uint32_t c = 0; c < primal.cell_count; ++c)
        for (auto i = primal.cell_offsets[c]; i < primal.cell_offsets[c + 1]; ++i) {
            const auto edge = primal.logical_edge_ordinals[i], gi = primal.gate_value_indices[i];
            if (!gates.admitted[gi]) continue;
            const auto s = e.source_indices[edge], d = e.destination_indices[edge];
            const float q = gates.probabilities[gi];
            const auto xi = (static_cast<std::uint64_t>(c) * source_count + s) * 64;
            const auto yi = (static_cast<std::uint64_t>(c) * destination_count + d) * 64;
            float dot = 0;
            for (std::uint32_t k = 0; k < 64; ++k) dot += dy[yi + k] * x[xi + k];
            dw[edge] += q * dot;
            for (std::uint32_t k = 0; k < 64; ++k) dx[xi + k] += weights[edge] * q * dy[yi + k];
        }
    std::uint64_t out = 0;
    for (std::uint32_t c = 0; c < derivative.cell_count; ++c)
        for (auto i = derivative.cell_offsets[c]; i < derivative.cell_offsets[c + 1]; ++i, ++out) {
            const auto edge = derivative.logical_edge_ordinals[i];
            const auto s = e.source_indices[edge], d = e.destination_indices[edge];
            const auto xi = (static_cast<std::uint64_t>(c) * source_count + s) * 64;
            const auto yi = (static_cast<std::uint64_t>(c) * destination_count + d) * 64;
            float dot = 0;
            for (std::uint32_t k = 0; k < 64; ++k) dot += dy[yi + k] * x[xi + k];
            // The caller applies the STE/sigmoid derivative. This is dL/dq for
            // each explicitly scored derivative candidate, admitted or dormant.
            dg[out] = weights[edge] * dot;
        }
    return status_v1::success;
}

status_v1 latent_transition_f32_reference_v1(
    latent_transition_shape_v1 s, std::span<const std::uint64_t> affected_offsets,
    std::span<const std::uint32_t> affected,
    std::span<const float> state, std::span<const float> input,
    std::span<const float> a, std::span<const float> b,
    std::span<const float> bias, std::span<float> next) noexcept {
    if (!valid_shape(s, affected_offsets, affected, state, input, a, b, bias, next))
        return status_v1::invalid_argument;
    std::copy(state.begin(), state.end(), next.begin());
    for (std::uint32_t c = 0; c < s.cells; ++c)
        for (auto ai = affected_offsets[c]; ai < affected_offsets[c + 1]; ++ai) {
            const auto m = affected[ai];
            for (std::uint32_t o = 0; o < s.state_width; ++o) {
                float v = bias[o];
                for (std::uint32_t i = 0; i < s.state_width; ++i)
                    v += a[static_cast<std::uint64_t>(o) * s.state_width + i]
                        * state[(static_cast<std::uint64_t>(c) * s.mechanisms + m) * s.state_width + i];
                for (std::uint32_t i = 0; i < s.input_width; ++i)
                    v += b[static_cast<std::uint64_t>(o) * s.input_width + i]
                        * input[static_cast<std::uint64_t>(c) * s.input_width + i];
                next[(static_cast<std::uint64_t>(c) * s.mechanisms + m) * s.state_width + o] = std::tanh(v);
            }
        }
    return status_v1::success;
}

status_v1 latent_transition_vjp_f32_reference_v1(
    latent_transition_shape_v1 s, std::span<const std::uint64_t> affected_offsets,
    std::span<const std::uint32_t> affected,
    std::span<const float> state, std::span<const float> input,
    std::span<const float> a, std::span<const float> b,
    std::span<const float> bias, std::span<const float> next,
    std::span<const float> next_grad, std::span<float> state_grad,
    std::span<float> input_grad, std::span<float> a_grad,
    std::span<float> b_grad, std::span<float> bias_grad) noexcept {
    if (!valid_shape(s, affected_offsets, affected, state, input, a, b, bias, next)
        || next_grad.size() != next.size() || state_grad.size() != state.size()
        || input_grad.size() != input.size() || a_grad.size() != a.size()
        || b_grad.size() != b.size() || bias_grad.size() != bias.size())
        return status_v1::invalid_argument;
    std::copy(next_grad.begin(), next_grad.end(), state_grad.begin()); // identity path for untouched mechanisms
    std::fill(input_grad.begin(), input_grad.end(), 0.0f);
    std::fill(a_grad.begin(), a_grad.end(), 0.0f);
    std::fill(b_grad.begin(), b_grad.end(), 0.0f);
    std::fill(bias_grad.begin(), bias_grad.end(), 0.0f);
    for (std::uint32_t c = 0; c < s.cells; ++c)
        for (auto ai = affected_offsets[c]; ai < affected_offsets[c + 1]; ++ai) {
            const auto m = affected[ai];
            const auto state_base = (static_cast<std::uint64_t>(c) * s.mechanisms + m)
                * s.state_width;
            std::fill(state_grad.begin() + state_base,
                      state_grad.begin() + state_base + s.state_width, 0.0f);
            for (std::uint32_t o = 0; o < s.state_width; ++o) {
                const auto ni = (static_cast<std::uint64_t>(c) * s.mechanisms + m) * s.state_width + o;
                const float dz = next_grad[ni] * (1.0f - next[ni] * next[ni]);
                bias_grad[o] += dz;
                for (std::uint32_t i = 0; i < s.state_width; ++i) {
                    const auto ai = static_cast<std::uint64_t>(o) * s.state_width + i;
                    const auto zi = (static_cast<std::uint64_t>(c) * s.mechanisms + m) * s.state_width + i;
                    a_grad[ai] += dz * state[zi];
                    state_grad[zi] += dz * a[ai];
                }
                for (std::uint32_t i = 0; i < s.input_width; ++i) {
                    const auto bi = static_cast<std::uint64_t>(o) * s.input_width + i;
                    const auto ui = static_cast<std::uint64_t>(c) * s.input_width + i;
                    b_grad[bi] += dz * input[ui];
                    input_grad[ui] += dz * b[bi];
                }
            }
        }
    return status_v1::success;
}

} // namespace cellerator::compute::operation::sparse_support
