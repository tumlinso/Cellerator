#pragma once

#include <Cellerator/compute/operation/indexed_mechanism/training.hh>
#include <Cellerator/compute/projection/physical_mma_hybrid.hh>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace cellerator::compute::operation::sparse_support {

enum class status_v1 : std::uint8_t {
    success, invalid_argument, insufficient_capacity, stale_epoch,
    invalid_projection, cuda_failure
};

struct parameter_block_v1 {
    std::string name;
    std::uint64_t offset = 0;
    std::vector<std::uint64_t> shape;
    std::uint64_t numel = 0;
};

// A sparse owner is a named flat FP32 master parameter vector. Its identity
// axis is a parameter-block axis; logical interaction edges never own optimizer
// parameters. The indexed-mechanism owner supplies reader/writer fencing.
class sparse_parameter_owner_v1 {
public:
    sparse_parameter_owner_v1(indexed::indexed_axis parameter_axis,
                              std::vector<indexed::identity> block_ids,
                              std::span<const float> initial,
                              std::vector<parameter_block_v1> blocks,
                              int device, std::uint32_t reader_capacity = 32);
    const std::shared_ptr<indexed::mechanism_parameter_owner>& native_owner() const noexcept;
    const std::vector<parameter_block_v1>& blocks() const noexcept;
    std::uint64_t parameter_count() const noexcept;
private:
    std::shared_ptr<indexed::mechanism_parameter_owner> owner_;
    std::vector<parameter_block_v1> blocks_;
};

struct edge_axis_view_v1 {
    std::uint64_t structure_epoch = 0;
    std::uint64_t edge_count = 0;
    // Arrays are in canonical logical-edge order, with IDs strictly increasing.
    const std::uint64_t* logical_edge_ids = nullptr;
    const std::uint32_t* source_indices = nullptr;
    const std::uint32_t* destination_indices = nullptr;
};

struct per_cell_edge_workset_view_v1 {
    std::uint64_t structure_epoch = 0;
    std::uint64_t state_generation = 0;
    std::uint32_t cell_count = 0;
    const std::uint64_t* cell_offsets = nullptr; // cell_count + 1
    const std::uint64_t* logical_edge_ordinals = nullptr;
    const std::uint64_t* logical_edge_ids = nullptr;
    // Direct index into edge_gate_values_view_v1 arrays, emitted with this
    // workset; avoids edge-ID searches in a launch.
    const std::uint64_t* gate_value_indices = nullptr;
    std::uint64_t item_count = 0;
    std::uint64_t capacity = 0;
};

// Probabilities/admission are aligned with a scored candidate list, which may
// include dormant candidates absent from the primal workset.
struct edge_gate_values_view_v1 {
    std::uint64_t structure_epoch = 0;
    std::uint64_t state_generation = 0;
    std::uint32_t cell_count = 0;
    const std::uint64_t* logical_edge_ordinals = nullptr;
    const std::uint64_t* logical_edge_ids = nullptr;
    const float* probabilities = nullptr;
    const std::uint8_t* admitted = nullptr;
    std::uint64_t candidate_count = 0;
    std::uint64_t capacity = 0;
};

// Independent host oracle. Inputs are row-major [cell, entity, 64], weights
// are logical-edge order, gates are dense [cell, edge] for the bounded oracle.
status_v1 forward_f32_reference_v1(
    const edge_axis_view_v1& edges, std::uint32_t source_count,
    std::uint32_t destination_count, std::uint32_t cell_count,
    std::span<const float> input, std::span<const float> weights,
    std::span<const float> gates, std::span<float> output) noexcept;

status_v1 backward_f32_reference_v1(
    const edge_axis_view_v1& edges, std::uint32_t source_count,
    std::uint32_t destination_count, std::uint32_t cell_count,
    std::span<const float> input, std::span<const float> weights,
    std::span<const float> gates, std::span<const float> output_gradient,
    std::span<float> input_gradient, std::span<float> weight_gradient,
    std::span<float> gate_gradient) noexcept;

status_v1 sparse_forward_f32_reference_v1(
    const edge_axis_view_v1& edges, const per_cell_edge_workset_view_v1& primal,
    const edge_gate_values_view_v1& gates, std::uint32_t source_count,
    std::uint32_t destination_count, std::span<const float> input,
    std::span<const float> weights, std::span<float> output) noexcept;

status_v1 sparse_backward_f32_reference_v1(
    const edge_axis_view_v1& edges, const per_cell_edge_workset_view_v1& primal,
    const per_cell_edge_workset_view_v1& derivative,
    const edge_gate_values_view_v1& gates, std::uint32_t source_count,
    std::uint32_t destination_count, std::span<const float> input,
    std::span<const float> weights, std::span<const float> output_gradient,
    std::span<float> input_gradient, std::span<float> weight_gradient,
    std::span<float> candidate_gate_gradient) noexcept;

struct relation_workspace_v1 {
    std::uint32_t max_cells = 0;
    std::uint64_t max_primal_items = 0;
    std::uint64_t max_derivative_items = 0;
    std::uint64_t max_edges = 0;
    std::uint32_t source_count = 0;
    std::uint32_t destination_count = 0;
    std::uint32_t dense_width = 64;
    std::uint64_t reserved_bytes = 0;
};

// Physical batch view compiled from CE-GEO's immutable N64 hybrid image. The
// value-map indices and per-edge tile-work indices are bound once per support
// projection; per-cell gate changes only alter the active workset and values.
struct ce_geo_n64_workset_view_v1 {
    std::uint64_t structure_epoch = 0;
    std::uint64_t state_generation = 0;
    std::uint64_t projection_id = 0;
    std::uint32_t cell_count = 0;
    std::uint32_t destination_group_count = 0;
    // Upper bound reserved at preparation; offsets' final element is the
    // run-time active count. Using the capacity lets steady execution avoid a
    // device-to-host count read before invoking N64.
    std::uint64_t active_tile_capacity = 0;
    std::uint32_t* cell_group_offsets = nullptr; // cells*groups + 1
    std::uint32_t* active_tile_ids = nullptr;    // CE-GEO tile indices
    std::uint32_t* active_tile_source_bases = nullptr; // includes cell offset
    std::uint32_t* item_tile_work_indices = nullptr;   // aligned to primal
    const std::uint32_t* destination_members = nullptr; // groups*16, invalid = UINT32_MAX
    const std::uint32_t* logical_source_to_physical = nullptr;
};

struct ce_geo_n64_workspace_v1 {
    std::uint64_t active_tile_capacity = 0;
    std::uint64_t primal_item_capacity = 0;
    std::uint64_t residual_item_capacity = 0;
    std::uint64_t physical_output_capacity = 0;
    std::uint64_t reserved_bytes = 0;
};

struct ce_geo_n64_compaction_workspace_v1 {
    std::uint64_t max_items = 0;
    std::uint64_t* unsorted_keys = nullptr;
    std::uint64_t* sorted_keys = nullptr;
    std::uint64_t* unique_keys = nullptr;
    std::uint32_t* unique_flags = nullptr;
    std::uint32_t* unique_prefix = nullptr;
    void* temporary_storage = nullptr;
    std::uint64_t temporary_storage_bytes = 0;
};

struct ce_geo_n64_projection_binding_v1 {
    std::uint64_t structure_epoch = 0;
    std::uint64_t projection_id = 0;
    projection::physical_mma_hybrid_header_v1 header{};
    // Projection-local logical edge IDs in support-ordinal order. These may
    // differ from stable support identities while value-map indices remain
    // aligned to the same support ordinals.
    std::vector<std::uint64_t> projection_edge_ids;
    // Direct support-ordinal -> CE-GEO value-map index; UINT64_MAX marks an
    // edge outside this stable projection (for example, a delta edge).
    std::vector<std::uint64_t> edge_value_map_indices;
    std::vector<std::uint32_t> tile_destination_groups;
    std::vector<std::uint32_t> tile_source_bases;
    // Physical source order and inverse destination scatter, each with 16
    // slots per group; padding slots use UINT32_MAX.
    std::vector<std::uint32_t> source_members;
    std::vector<std::uint32_t> destination_members;
    std::vector<std::uint32_t> logical_source_to_physical;
};

status_v1 prepare_ce_geo_n64_projection_v1(
    const void* ce_geo_image, std::size_t image_bytes,
    // Projection-local edge IDs in canonical support-ordinal order. They may
    // differ from the stable IDs in edge_axis_view_v1.
    std::span<const std::uint64_t> projection_edge_ids_in_support_order,
    std::span<const std::uint32_t> support_source_indices,
    std::span<const std::uint32_t> support_destination_indices,
    std::span<const std::uint32_t> source_group_members,
    std::span<const std::uint32_t> destination_group_members,
    std::uint32_t logical_source_count, std::uint32_t logical_destination_count,
    std::uint64_t structure_epoch, std::uint64_t projection_id,
    ce_geo_n64_projection_binding_v1* output) noexcept;

std::uint64_t ce_geo_n64_compaction_temporary_bytes_v1(
    std::uint64_t max_items) noexcept;

status_v1 prepare_ce_geo_n64_workset_cuda_v1(
    const projection::physical_mma_hybrid_header_v1& projection_header,
    const edge_axis_view_v1& edges,
    const per_cell_edge_workset_view_v1& admitted_mma_workset,
    const edge_gate_values_view_v1& gates,
    const std::uint64_t* edge_value_map_indices,
    const std::uint64_t* projection_edge_ids,
    const projection::projection_value_map_v1* value_maps,
    const std::uint32_t* tile_destination_groups,
    const std::uint32_t* tile_source_bases,
    std::uint32_t physical_source_count,
    ce_geo_n64_workset_view_v1* output_workset,
    ce_geo_n64_compaction_workspace_v1 workspace,
    void* caller_stream) noexcept;

// CUDA interface uses FP16 input, weight and admitted gate multiplication with
// FP32 output/accumulation/gradients. All buffers and the explicit caller stream
// are caller-owned. Worksets must be prepared, epoch-matched and in capacity.
// Width 64 is the initial CE-GEO packed anchor; SIMT is the irregular fallback.
status_v1 relation_forward_cuda_v1(
    const edge_axis_view_v1& edges, const per_cell_edge_workset_view_v1& primal,
    const edge_gate_values_view_v1& gates, const std::uint16_t* input_half,
    std::uint32_t source_count, std::uint32_t destination_count,
    const std::uint16_t* logical_weights_half, float* output,
    void* caller_stream) noexcept;

status_v1 relation_backward_cuda_v1(
    const edge_axis_view_v1& edges, const per_cell_edge_workset_view_v1& primal,
    const per_cell_edge_workset_view_v1& derivative,
    const edge_gate_values_view_v1& gates, const std::uint16_t* input_half,
    std::uint32_t source_count, std::uint32_t destination_count,
    const std::uint16_t* logical_weights_half, const float* output_gradient,
    float* input_gradient, float* logical_weight_gradient,
    float* candidate_gate_gradient, void* caller_stream) noexcept;

// Executes CE-GEO's actual N64 tile cover through its existing Volta WMMA
// consumer. Inactive tiles are omitted from compact per-cell destination lists;
// derivative work remains an independent SIMT candidate set. Residual-covered
// edges are supplied separately and accumulate into the logical output.
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
    const ce_geo_n64_workspace_v1& reserved_workspace,
    const std::uint16_t* input_half, std::uint32_t logical_source_count,
    std::uint32_t physical_source_count,
    std::uint32_t destination_count, const std::uint16_t* logical_weights_half,
    std::uint16_t* active_tile_values, float* physical_output,
    float* logical_output, void* caller_stream) noexcept;

// Continually discovered/revoked support remains a sparse SIMT delta until
// publication installs a replacement CE-GEO projection. This adds edges with
// no base value-map binding into an already initialized logical output.
status_v1 relation_delta_accumulate_ce_geo_n64_cuda_v1(
    std::uint64_t value_map_count,
    const std::uint64_t* edge_value_map_indices,
    const edge_axis_view_v1& edges,
    const per_cell_edge_workset_view_v1& delta_primal,
    const edge_gate_values_view_v1& gates,
    const std::uint16_t* logical_input_half,
    std::uint32_t logical_source_count,
    std::uint32_t destination_count, const std::uint16_t* logical_weights_half,
    float* logical_output, void* caller_stream) noexcept;

status_v1 gather_source_to_ce_geo_n64_cuda_v1(
    const std::uint32_t* physical_source_members,
    std::uint32_t physical_source_count, std::uint32_t logical_source_count,
    std::uint32_t cell_count, const std::uint16_t* logical_input_half,
    std::uint16_t* physical_input_half, void* caller_stream) noexcept;

// Shared-parameter latent transition z' = tanh(Az + Bu + b). State is
// row-major [cell, mechanism, width]. The affected mechanism list may overlap
// across entities; each mechanism is emitted once per cell in the workset.
struct latent_transition_shape_v1 {
    std::uint32_t cells = 0;
    std::uint32_t mechanisms = 0;
    std::uint32_t state_width = 0;
    std::uint32_t input_width = 0;
};

status_v1 latent_transition_cuda_v1(
    latent_transition_shape_v1 shape, const std::uint64_t* affected_offsets,
    const std::uint32_t* affected_mechanisms, std::uint64_t affected_count,
    const std::uint16_t* state_half,
    const std::uint16_t* input_half, const std::uint16_t* matrix_a_half,
    const std::uint16_t* matrix_b_half, const std::uint16_t* bias_half,
    float* next_state, void* caller_stream) noexcept;

status_v1 latent_transition_vjp_cuda_v1(
    latent_transition_shape_v1 shape, const std::uint64_t* affected_offsets,
    const std::uint32_t* affected_mechanisms, std::uint64_t affected_count,
    const std::uint16_t* state_half,
    const std::uint16_t* input_half, const std::uint16_t* matrix_a_half,
    const std::uint16_t* matrix_b_half, const float* next_state,
    const float* next_state_gradient, float* state_gradient,
    float* input_gradient, float* matrix_a_gradient,
    float* matrix_b_gradient, float* bias_gradient,
    void* caller_stream) noexcept;

status_v1 latent_transition_f32_reference_v1(
    latent_transition_shape_v1 shape, std::span<const std::uint64_t> affected_offsets,
    std::span<const std::uint32_t> affected_mechanisms,
    std::span<const float> state, std::span<const float> input,
    std::span<const float> matrix_a, std::span<const float> matrix_b,
    std::span<const float> bias, std::span<float> next_state) noexcept;

status_v1 latent_transition_vjp_f32_reference_v1(
    latent_transition_shape_v1 shape, std::span<const std::uint64_t> affected_offsets,
    std::span<const std::uint32_t> affected_mechanisms,
    std::span<const float> state, std::span<const float> input,
    std::span<const float> matrix_a, std::span<const float> matrix_b,
    std::span<const float> bias, std::span<const float> next_state,
    std::span<const float> next_gradient, std::span<float> state_gradient,
    std::span<float> input_gradient, std::span<float> matrix_a_gradient,
    std::span<float> matrix_b_gradient, std::span<float> bias_gradient) noexcept;

} // namespace cellerator::compute::operation::sparse_support
