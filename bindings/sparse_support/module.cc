#include "module.hh"

#include "geometry_bindings.hh"
#include "indexing_bindings.hh"

#include <Cellerator/compute/operation/sparse_support/training.hh>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/torch.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace py = pybind11;
namespace ss = cellerator::compute::operation::sparse_support;
namespace ix = cellerator::compute::operation::indexed;
namespace sb = cellerator::bindings::sparse_support;

namespace cellerator::bindings::sparse_support {

struct prepared_ce_geo_n64_torch_v1 {
    ss::ce_geo_n64_projection_binding_v1 binding;
    std::vector<std::uint8_t> image_bytes;
    std::vector<std::uint32_t> source_group_members_host;
    std::vector<std::uint32_t> destination_group_members_host;
    torch::Tensor image;
    torch::Tensor edge_value_map_indices;
    torch::Tensor projection_edge_ids;
    std::vector<std::uint64_t> projection_edge_ids_host;
    torch::Tensor tile_destination_groups;
    torch::Tensor tile_source_bases;
    torch::Tensor source_members;
    torch::Tensor destination_members;
    torch::Tensor logical_source_to_physical;
    std::uint32_t max_cells = 0;
    std::uint64_t max_primal_items = 0;
    std::uint64_t projection_id = 0;
    cudaEvent_t ready_event = nullptr;
    std::mutex workspaces_mutex;
    std::unordered_map<std::uintptr_t, std::shared_ptr<struct ce_geo_work_buffers_v1>> workspaces;
    ~prepared_ce_geo_n64_torch_v1() {
        if (ready_event) cudaEventDestroy(ready_event);
    }
    void record_ready(cudaStream_t stream) {
        TORCH_CHECK(cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming) == cudaSuccess,
                    "could not create CE-GEO projection readiness event");
        TORCH_CHECK(cudaEventRecord(ready_event, stream) == cudaSuccess,
                    "could not record CE-GEO projection readiness event");
    }
    void wait_ready(cudaStream_t stream) const {
        if (ready_event)
            TORCH_CHECK(cudaStreamWaitEvent(stream, ready_event, 0) == cudaSuccess,
                        "could not order caller stream after CE-GEO preparation");
    }
};

struct ce_geo_work_buffers_v1 {
    torch::Tensor cell_group_offsets;
    torch::Tensor active_tile_ids;
    torch::Tensor active_tile_source_bases;
    torch::Tensor item_tile_work_indices;
    torch::Tensor active_tile_values;
    torch::Tensor physical_output;
    torch::Tensor physical_input;
    torch::Tensor unsorted_keys;
    torch::Tensor sorted_keys;
    torch::Tensor unique_keys;
    torch::Tensor unique_flags;
    torch::Tensor unique_prefix;
    torch::Tensor temporary;
    torch::Tensor logical_weights_half;
    std::uint64_t max_active_tiles = 0;
    std::uint64_t temporary_bytes = 0;
};

class sparse_torch_handle_v1 final {
public:
    sparse_torch_handle_v1(std::shared_ptr<native_support_handle_v1> support,
            std::shared_ptr<ss::sparse_parameter_owner_v1> parameters,
            std::vector<std::uint64_t> edge_ids,
            std::vector<std::uint32_t> sources,
            std::vector<std::uint32_t> destinations,
            std::uint32_t source_count, std::uint32_t destination_count,
            std::uint64_t structure_epoch, int device)
        : support_(std::move(support)), parameters_(std::move(parameters)),
          edge_ids_(std::move(edge_ids)), sources_(std::move(sources)),
          destinations_(std::move(destinations)), source_count_(source_count),
          destination_count_(destination_count), structure_epoch_(structure_epoch),
          device_(device) {}

    const std::shared_ptr<ss::sparse_parameter_owner_v1>& parameters() const noexcept {
        return parameters_;
    }
    const std::shared_ptr<native_support_handle_v1>& support() const noexcept {
        return support_;
    }
    const std::vector<std::uint64_t>& edge_ids() const noexcept { return edge_ids_; }
    const torch::Tensor& edge_ids_device() const noexcept { return edge_ids_device_; }
    const std::vector<std::uint32_t>& sources() const noexcept { return sources_; }
    const std::vector<std::uint32_t>& destinations() const noexcept { return destinations_; }
    const torch::Tensor& source_indices_device() const noexcept { return source_indices_device_; }
    const torch::Tensor& destination_indices_device() const noexcept { return destination_indices_device_; }
    void set_endpoint_tensors(torch::Tensor ids, torch::Tensor sources, torch::Tensor destinations) {
        edge_ids_device_ = std::move(ids);
        source_indices_device_ = std::move(sources);
        destination_indices_device_ = std::move(destinations);
    }
    void record_endpoints_ready(cudaStream_t stream) {
        TORCH_CHECK(cudaEventCreateWithFlags(&endpoints_ready_event_, cudaEventDisableTiming)
                        == cudaSuccess,
                    "could not create sparse endpoint readiness event");
        TORCH_CHECK(cudaEventRecord(endpoints_ready_event_, stream) == cudaSuccess,
                    "could not record sparse endpoint readiness event");
    }
    void wait_endpoints_ready(cudaStream_t stream) const {
        if (endpoints_ready_event_)
            TORCH_CHECK(cudaStreamWaitEvent(stream, endpoints_ready_event_, 0) == cudaSuccess,
                        "could not order caller stream after sparse endpoint preparation");
    }
    std::uint32_t source_count() const noexcept { return source_count_; }
    std::uint32_t destination_count() const noexcept { return destination_count_; }
    std::uint64_t structure_epoch() const noexcept { return structure_epoch_; }
    int device() const noexcept { return device_; }
    std::shared_ptr<void> binding_adapter_state() const {
        std::lock_guard<std::mutex> lock(adapter_mutex_);
        return adapter_state_;
    }
    void set_binding_adapter_state(std::shared_ptr<void> state) {
        std::lock_guard<std::mutex> lock(adapter_mutex_);
        adapter_state_ = std::move(state);
    }
    std::shared_ptr<prepared_ce_geo_n64_torch_v1> ce_geo_projection() const {
        std::lock_guard<std::mutex> lock(projection_mutex_);
        return projection_;
    }
    void set_ce_geo_projection(std::shared_ptr<prepared_ce_geo_n64_torch_v1> projection) {
        std::lock_guard<std::mutex> lock(projection_mutex_);
        projection_ = std::move(projection);
    }

private:
    std::shared_ptr<native_support_handle_v1> support_;
    std::shared_ptr<ss::sparse_parameter_owner_v1> parameters_;
    std::vector<std::uint64_t> edge_ids_;
    std::vector<std::uint32_t> sources_, destinations_;
    torch::Tensor edge_ids_device_;
    torch::Tensor source_indices_device_, destination_indices_device_;
    std::uint32_t source_count_ = 0, destination_count_ = 0;
    std::uint64_t structure_epoch_ = 0;
    int device_ = -1;
    cudaEvent_t endpoints_ready_event_ = nullptr;
    mutable std::mutex adapter_mutex_;
    std::shared_ptr<void> adapter_state_;
    mutable std::mutex projection_mutex_;
    std::shared_ptr<prepared_ce_geo_n64_torch_v1> projection_;
public:
    ~sparse_torch_handle_v1() {
        if (endpoints_ready_event_) cudaEventDestroy(endpoints_ready_event_);
    }
};

} // namespace cellerator::bindings::sparse_support

namespace {

using SparseHandle = sb::sparse_torch_handle_v1;
using SparseHandlePtr = std::shared_ptr<SparseHandle>;

std::uintptr_t tensor_version(const torch::Tensor& tensor) {
    return static_cast<std::uintptr_t>(
        tensor.unsafeGetTensorImpl()->version_counter().current_version());
}

void check_status(ss::status_v1 status, const char* operation) {
    TORCH_CHECK(status == ss::status_v1::success, operation,
                " failed with native sparse support status ",
                static_cast<int>(status));
}

struct SparseAliasState {
    torch::Tensor tensor;
    std::vector<std::pair<torch::Tensor, std::uintptr_t>> aliases;
    std::mutex mutex;
};

std::mutex registry_mutex;
std::unordered_map<void*, std::weak_ptr<ix::mechanism_parameter_owner>> owner_registry;

std::shared_ptr<SparseAliasState> alias_state(const SparseHandlePtr& handle) {
    TORCH_CHECK(handle, "sparse support handle is null");
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto state = std::static_pointer_cast<SparseAliasState>(handle->binding_adapter_state());
    if (!state) {
        state = std::make_shared<SparseAliasState>();
        handle->set_binding_adapter_state(state);
    }
    return state;
}

// Handle-local adapter state uses the native support manager's opaque adapter
// slot, which is otherwise unused by this owner. Keep the handle Torch-free.

std::vector<std::int64_t> block_metadata(const SparseHandlePtr& handle) {
    std::vector<std::int64_t> result;
    for (const auto& block : handle->parameters()->blocks()) {
        result.push_back(static_cast<std::int64_t>(block.offset));
        result.push_back(static_cast<std::int64_t>(block.numel));
        result.push_back(static_cast<std::int64_t>(block.shape.size()));
        for (const auto dim : block.shape) result.push_back(static_cast<std::int64_t>(dim));
    }
    return result;
}

void validate_parameters_locked(const torch::Tensor& tensor,
                                const std::shared_ptr<SparseAliasState>& state,
                                const SparseHandlePtr& handle) {
    const auto& owner = handle->parameters()->native_owner();
    TORCH_CHECK(!owner->poisoned(), "sparse parameter owner is poisoned");
    TORCH_CHECK(tensor.defined() && tensor.is_cuda()
            && tensor.scalar_type() == torch::kFloat32 && tensor.dim() == 1
            && tensor.is_contiguous() && tensor.get_device() == owner->device()
            && tensor.numel() == static_cast<std::int64_t>(owner->size()),
        "sparse parameter tensor metadata no longer matches canonical FP32 storage");
    TORCH_CHECK(tensor.data_ptr<float>() == owner->data(),
        "sparse parameter tensor no longer aliases canonical native storage");
    for (const auto& alias : state->aliases)
        TORCH_CHECK(tensor_version(alias.first) == alias.second,
            "sparse parameters changed outside guarded_step");
    auto found = std::find_if(state->aliases.begin(), state->aliases.end(),
        [&](const auto& alias) {
            return alias.first.unsafeGetTensorImpl() == tensor.unsafeGetTensorImpl();
        });
    if (found == state->aliases.end()) state->aliases.emplace_back(tensor, tensor_version(tensor));
    else TORCH_CHECK(tensor_version(tensor) == found->second,
        "sparse parameter tensor version changed outside guarded_step");
}

void validate_dense_cuda(const torch::Tensor& tensor, const char* name,
                         torch::ScalarType type, int device, std::int64_t rank) {
    TORCH_CHECK(tensor.defined() && tensor.is_cuda() && tensor.get_device() == device,
                name, " must reside on the sparse owner's CUDA device");
    TORCH_CHECK(tensor.scalar_type() == type && tensor.dim() == rank && tensor.is_contiguous(),
                name, " has invalid dtype, rank, or contiguity");
}

template<class T>
torch::Tensor copy_vector_to_device(const std::vector<T>& values,
                                    torch::ScalarType type, int device) {
    auto host = torch::empty({static_cast<std::int64_t>(values.size())},
        torch::TensorOptions().dtype(type).device(torch::kCPU));
    if (!values.empty())
        std::memcpy(host.data_ptr(), values.data(), values.size() * sizeof(T));
    return host.to(torch::Device(torch::kCUDA, device));
}

std::shared_ptr<sb::ce_geo_work_buffers_v1> ce_geo_work_buffers(
        const std::shared_ptr<sb::prepared_ce_geo_n64_torch_v1>& packed,
        std::uint32_t cells, at::cuda::CUDAStream stream) {
    const auto key = reinterpret_cast<std::uintptr_t>(stream.stream());
    std::lock_guard<std::mutex> lock(packed->workspaces_mutex);
    auto& buffers = packed->workspaces[key];
    if (!buffers) {
        const auto device = packed->image.get_device();
        const auto groups = packed->binding.header.destination_group_count;
        const auto active = std::min<std::uint64_t>(packed->max_primal_items,
            static_cast<std::uint64_t>(packed->max_cells) * packed->binding.header.tile_count);
        auto options = torch::TensorOptions().device(torch::Device(torch::kCUDA, device));
        auto i32 = options.dtype(torch::kInt32);
        auto u64 = options.dtype(torch::kUInt64);
        auto f16 = options.dtype(torch::kFloat16);
        auto f32 = options.dtype(torch::kFloat32);
        auto bytes = options.dtype(torch::kUInt8);
        buffers = std::make_shared<sb::ce_geo_work_buffers_v1>();
        buffers->max_active_tiles = active;
        buffers->temporary_bytes = ss::ce_geo_n64_compaction_temporary_bytes_v1(
            packed->max_primal_items);
        buffers->cell_group_offsets = torch::empty(
            {static_cast<std::int64_t>(packed->max_cells) * groups + 1}, i32);
        buffers->active_tile_ids = torch::empty({static_cast<std::int64_t>(active)}, i32);
        buffers->active_tile_source_bases = torch::empty({static_cast<std::int64_t>(active)}, i32);
        buffers->item_tile_work_indices = torch::empty(
            {static_cast<std::int64_t>(packed->max_primal_items)}, i32);
        buffers->active_tile_values = torch::empty({static_cast<std::int64_t>(active) * 256}, f16);
        buffers->physical_output = torch::empty(
            {static_cast<std::int64_t>(packed->max_cells) * groups * 16 * 64}, f32);
        buffers->physical_input = torch::empty(
            {static_cast<std::int64_t>(packed->max_cells)
                * static_cast<std::int64_t>(packed->binding.source_members.size()) * 64}, f16);
        buffers->logical_weights_half = torch::empty(
            {static_cast<std::int64_t>(packed->binding.edge_value_map_indices.size())}, f16);
        buffers->unsorted_keys = torch::empty(
            {static_cast<std::int64_t>(packed->max_primal_items)}, u64);
        buffers->sorted_keys = torch::empty_like(buffers->unsorted_keys);
        buffers->unique_keys = torch::empty_like(buffers->unsorted_keys);
        buffers->unique_flags = torch::empty(
            {static_cast<std::int64_t>(packed->max_primal_items)}, i32);
        buffers->unique_prefix = torch::empty_like(buffers->unique_flags);
        buffers->temporary = torch::empty({static_cast<std::int64_t>(
            std::max<std::uint64_t>(buffers->temporary_bytes, 1))}, bytes);
    }
    TORCH_CHECK(cells <= packed->max_cells,
                "batch exceeds the CE-GEO workspace capacity for this stream");
    return buffers;
}

ss::edge_axis_view_v1 edge_view(const SparseHandlePtr& handle) {
    return {handle->structure_epoch(), handle->edge_ids().size(),
        reinterpret_cast<const std::uint64_t*>(handle->edge_ids_device().data_ptr()),
        reinterpret_cast<const std::uint32_t*>(handle->source_indices_device().data_ptr()),
        reinterpret_cast<const std::uint32_t*>(handle->destination_indices_device().data_ptr())};
}

ss::per_cell_edge_workset_view_v1 workset_view(
        const torch::Tensor& offsets, const torch::Tensor& ordinals,
        const torch::Tensor& ids, const torch::Tensor& gate_indices,
        std::uint64_t state_generation, std::uint64_t capacity,
        std::uint64_t epoch, std::uint32_t cells) {
    return {epoch, state_generation, cells,
        reinterpret_cast<const std::uint64_t*>(offsets.data_ptr<std::int64_t>()),
        reinterpret_cast<const std::uint64_t*>(ordinals.data_ptr<std::int64_t>()),
        reinterpret_cast<const std::uint64_t*>(ids.data_ptr<std::int64_t>()),
        reinterpret_cast<const std::uint64_t*>(gate_indices.data_ptr<std::int64_t>()),
        static_cast<std::uint64_t>(ordinals.numel()), capacity};
}

ss::edge_gate_values_view_v1 gate_view(
        const torch::Tensor& ids, const torch::Tensor& ordinals,
        const torch::Tensor& probabilities, const torch::Tensor& admitted,
        std::uint64_t state_generation, std::uint64_t epoch,
        std::uint32_t cells) {
    return {epoch, state_generation, cells,
        reinterpret_cast<const std::uint64_t*>(ordinals.data_ptr<std::int64_t>()),
        reinterpret_cast<const std::uint64_t*>(ids.data_ptr<std::int64_t>()), probabilities.data_ptr<float>(),
        admitted.data_ptr<std::uint8_t>(),
        static_cast<std::uint64_t>(probabilities.numel()),
        static_cast<std::uint64_t>(probabilities.numel())};
}

struct SparseHandleCapsule final : torch::CustomClassHolder {
    explicit SparseHandleCapsule(SparseHandlePtr value) : handle(std::move(value)) {}
    SparseHandlePtr handle;
};

struct SparseTape final : torch::CustomClassHolder {
    SparseHandlePtr handle;
    std::uint64_t epoch = 0;
    std::atomic<bool> pinned{false};
    std::atomic<bool> consumed{false};
    std::atomic<bool> retired{false};
    void retire() noexcept {
        if (pinned.load(std::memory_order_acquire)
            && !retired.exchange(true, std::memory_order_acq_rel) && handle)
            (void)handle->support()->retire_forward_tape(epoch);
    }
    ~SparseTape() override { retire(); }
};

torch::Tensor sparse_relation_forward_packed(
        const SparseHandlePtr& handle, const std::shared_ptr<sb::prepared_ce_geo_n64_torch_v1>& packed,
        const torch::Tensor& input, const torch::Tensor& edge_values,
        const torch::Tensor& probabilities, const torch::Tensor& admitted,
        const torch::Tensor& primal_offsets, const torch::Tensor& primal_ordinals,
        const torch::Tensor& primal_ids, const torch::Tensor& primal_gate_indices,
        const torch::Tensor& gate_ids, const torch::Tensor& gate_ordinals,
        std::uint64_t primal_state_generation, std::uint64_t primal_capacity,
        std::uint64_t gate_state_generation, std::uint64_t epoch,
        std::uint32_t cells, at::cuda::CUDAStream stream) {
    TORCH_CHECK(packed->binding.structure_epoch == epoch,
                "CE-GEO projection structure epoch is stale");
    TORCH_CHECK(cells <= packed->max_cells && primal_capacity <= packed->max_primal_items,
                "batch or workset exceeds the prepared CE-GEO capacity");
    const auto& header = packed->binding.header;
    packed->wait_ready(stream.stream());
    const std::uint64_t max_active_tiles = std::min<std::uint64_t>(
        primal_capacity, static_cast<std::uint64_t>(cells) * header.tile_count);
    const auto destination_groups = header.destination_group_count;
    const auto physical_sources = static_cast<std::uint32_t>(
        packed->binding.source_members.size());
    auto buffers = ce_geo_work_buffers(packed, cells, stream);
    TORCH_CHECK(max_active_tiles <= buffers->max_active_tiles,
                "active tile capacity exceeds the reserved CE-GEO workspace");
    auto& cell_group_offsets = buffers->cell_group_offsets;
    auto& active_tile_ids = buffers->active_tile_ids;
    auto& active_tile_source_bases = buffers->active_tile_source_bases;
    auto& item_tile_work_indices = buffers->item_tile_work_indices;
    auto& active_tile_values = buffers->active_tile_values;
    auto& physical_output = buffers->physical_output;
    auto& physical_input = buffers->physical_input;
    auto& logical_weights_half = buffers->logical_weights_half;
    auto& unsorted_keys = buffers->unsorted_keys;
    auto& sorted_keys = buffers->sorted_keys;
    auto& unique_keys = buffers->unique_keys;
    auto& unique_flags = buffers->unique_flags;
    auto& unique_prefix = buffers->unique_prefix;
    auto& temporary = buffers->temporary;
    auto float_options = input.options().dtype(torch::kFloat32);
    physical_output.narrow(0, 0,
        static_cast<std::int64_t>(cells) * destination_groups * 16 * 64).zero_();
    auto output = torch::zeros({cells, handle->destination_count(), 64}, float_options);
    const auto temporary_bytes = buffers->temporary_bytes;

    const auto primal = workset_view(primal_offsets, primal_ordinals, primal_ids,
        primal_gate_indices, primal_state_generation, primal_capacity, epoch, cells);
    const auto gates = gate_view(gate_ids, gate_ordinals, probabilities, admitted,
        gate_state_generation, epoch, cells);
    ss::ce_geo_n64_workset_view_v1 physical_workset{};
    physical_workset.structure_epoch = epoch;
    physical_workset.state_generation = primal_state_generation;
    physical_workset.projection_id = packed->binding.projection_id;
    physical_workset.cell_count = cells;
    physical_workset.destination_group_count = destination_groups;
    physical_workset.active_tile_capacity = max_active_tiles;
    physical_workset.cell_group_offsets = reinterpret_cast<std::uint32_t*>(cell_group_offsets.data_ptr<std::int32_t>());
    physical_workset.active_tile_ids = reinterpret_cast<std::uint32_t*>(active_tile_ids.data_ptr<std::int32_t>());
    physical_workset.active_tile_source_bases = reinterpret_cast<std::uint32_t*>(active_tile_source_bases.data_ptr<std::int32_t>());
    physical_workset.item_tile_work_indices = reinterpret_cast<std::uint32_t*>(item_tile_work_indices.data_ptr<std::int32_t>());
    physical_workset.destination_members = reinterpret_cast<const std::uint32_t*>(packed->destination_members.data_ptr<std::int32_t>());
    physical_workset.logical_source_to_physical = reinterpret_cast<const std::uint32_t*>(packed->logical_source_to_physical.data_ptr<std::int32_t>());
    ss::ce_geo_n64_compaction_workspace_v1 compaction{};
    compaction.max_items = primal_capacity;
    compaction.unsorted_keys = reinterpret_cast<std::uint64_t*>(unsorted_keys.data_ptr());
    compaction.sorted_keys = reinterpret_cast<std::uint64_t*>(sorted_keys.data_ptr());
    compaction.unique_keys = reinterpret_cast<std::uint64_t*>(unique_keys.data_ptr());
    compaction.unique_flags = reinterpret_cast<std::uint32_t*>(unique_flags.data_ptr<std::int32_t>());
    compaction.unique_prefix = reinterpret_cast<std::uint32_t*>(unique_prefix.data_ptr<std::int32_t>());
    compaction.temporary_storage = temporary.data_ptr<std::uint8_t>();
    compaction.temporary_storage_bytes = temporary_bytes;
    const auto* image = packed->image.data_ptr<std::uint8_t>();
    const auto* value_maps = reinterpret_cast<const cellerator::compute::projection::projection_value_map_v1*>(
        image + header.value_map_offset);
    check_status(ss::prepare_ce_geo_n64_workset_cuda_v1(
        header, edge_view(handle), primal, gates,
        reinterpret_cast<const std::uint64_t*>(packed->edge_value_map_indices.data_ptr()),
        reinterpret_cast<const std::uint64_t*>(packed->projection_edge_ids.data_ptr()),
        value_maps,
        reinterpret_cast<const std::uint32_t*>(packed->tile_destination_groups.data_ptr<std::int32_t>()),
        reinterpret_cast<const std::uint32_t*>(packed->tile_source_bases.data_ptr<std::int32_t>()),
        physical_sources, &physical_workset, compaction, stream.stream()),
        "CE-GEO N64 workset preparation");
    check_status(ss::gather_source_to_ce_geo_n64_cuda_v1(
        reinterpret_cast<const std::uint32_t*>(packed->source_members.data_ptr<std::int32_t>()),
        physical_sources, handle->source_count(), cells,
        reinterpret_cast<const std::uint16_t*>(input.data_ptr<at::Half>()),
        reinterpret_cast<std::uint16_t*>(physical_input.data_ptr<at::Half>()), stream.stream()),
        "CE-GEO N64 source gather");
    logical_weights_half.copy_(edge_values);
    ss::ce_geo_n64_workspace_v1 reserved{
        max_active_tiles, primal_capacity, primal_capacity,
        static_cast<std::uint64_t>(cells) * destination_groups * 16 * 64, 0u};
    check_status(ss::relation_forward_ce_geo_n64_cuda_v1(
        header, value_maps, header.value_map_count,
        reinterpret_cast<const std::uint64_t*>(packed->edge_value_map_indices.data_ptr()),
        reinterpret_cast<const std::uint64_t*>(packed->projection_edge_ids.data_ptr()),
        edge_view(handle), primal, primal, gates, physical_workset, reserved,
        reinterpret_cast<const std::uint16_t*>(physical_input.data_ptr<at::Half>()),
        handle->source_count(), physical_sources, handle->destination_count(),
        reinterpret_cast<const std::uint16_t*>(logical_weights_half.data_ptr<at::Half>()),
        reinterpret_cast<std::uint16_t*>(active_tile_values.data_ptr<at::Half>()),
        physical_output.data_ptr<float>(), output.data_ptr<float>(), stream.stream()),
        "CE-GEO N64 packed relation forward");
    check_status(ss::relation_delta_accumulate_ce_geo_n64_cuda_v1(
        header.value_map_count,
        reinterpret_cast<const std::uint64_t*>(packed->edge_value_map_indices.data_ptr()),
        edge_view(handle), primal, gates,
        reinterpret_cast<const std::uint16_t*>(input.data_ptr<at::Half>()),
        handle->source_count(), handle->destination_count(),
        reinterpret_cast<const std::uint16_t*>(logical_weights_half.data_ptr<at::Half>()),
        output.data_ptr<float>(), stream.stream()), "CE-GEO support delta accumulation");
    for (const auto* tensor : {&cell_group_offsets, &active_tile_ids,
            &active_tile_source_bases, &item_tile_work_indices, &active_tile_values,
            &physical_output, &physical_input, &unsorted_keys, &sorted_keys,
            &unique_keys, &unique_flags, &unique_prefix, &temporary, &logical_weights_half,
            &packed->image, &packed->edge_value_map_indices,
            &packed->tile_destination_groups, &packed->tile_source_bases,
            &packed->source_members, &packed->destination_members,
            &packed->logical_source_to_physical})
        c10::cuda::CUDACachingAllocator::recordStream(tensor->storage().data_ptr(), stream);
    return output;
}

torch::Tensor sparse_relation_forward(torch::autograd::AutogradContext* ctx,
    torch::Tensor input, torch::Tensor edge_values, torch::Tensor probabilities,
    torch::Tensor admitted, torch::Tensor primal_offsets,
    torch::Tensor primal_ordinals, torch::Tensor primal_ids,
    torch::Tensor primal_gate_indices, torch::Tensor derivative_offsets,
    torch::Tensor derivative_ordinals, torch::Tensor derivative_ids,
    torch::Tensor derivative_gate_indices, torch::Tensor gate_ids,
    torch::Tensor gate_ordinals, std::uint64_t primal_state_generation,
    std::uint64_t primal_capacity, std::uint64_t derivative_state_generation,
    std::uint64_t derivative_capacity, std::uint64_t gate_state_generation,
    SparseHandlePtr handle) {
    TORCH_CHECK(handle, "sparse support handle is null");
    const auto device = handle->device();
    validate_dense_cuda(input, "input", torch::kFloat16, device, 3);
    validate_dense_cuda(edge_values, "edge_values", torch::kFloat32, device, 1);
    validate_dense_cuda(probabilities, "probabilities", torch::kFloat32, device, 1);
    validate_dense_cuda(admitted, "admitted", torch::kUInt8, device, 1);
    TORCH_CHECK(input.size(1) == handle->source_count() && input.size(2) == 64,
                "relation input shape must be [cells, sources, 64]");
    TORCH_CHECK(edge_values.numel() == static_cast<std::int64_t>(handle->edge_ids().size()),
                "logical edge value count does not match support snapshot");
    TORCH_CHECK(probabilities.numel() == admitted.numel()
                && probabilities.numel() == gate_ids.numel()
                && probabilities.numel() == gate_ordinals.numel(),
                "gate candidate arrays must have matching lengths");
    const auto cells = static_cast<std::uint32_t>(input.size(0));
    const auto epoch = handle->structure_epoch();
    for (const auto* tensor : {&primal_offsets, &primal_ordinals, &primal_ids,
            &primal_gate_indices, &derivative_offsets, &derivative_ordinals,
            &derivative_ids, &derivative_gate_indices, &gate_ids, &gate_ordinals})
        validate_dense_cuda(*tensor, "workset", torch::kInt64, device, 1);
    TORCH_CHECK(primal_offsets.numel() == cells + 1
                && derivative_offsets.numel() == cells + 1,
                "workset CSR offsets must have cells + 1 entries");
    TORCH_CHECK(primal_state_generation == derivative_state_generation
                && derivative_state_generation == gate_state_generation,
                "primal, derivative, and gate state generations differ");

    auto tape = c10::make_intrusive<SparseTape>();
    tape->handle = handle;
    tape->epoch = epoch;
    auto support_status = handle->support()->pin_forward_tape(epoch);
    TORCH_CHECK(support_status == cellerator::compute::support::adaptive_support_status_v1::success,
                "could not pin sparse support structure epoch");
    tape->pinned.store(true, std::memory_order_release);

    auto stream = at::cuda::getCurrentCUDAStream(device);
    const auto primal = workset_view(primal_offsets, primal_ordinals, primal_ids,
        primal_gate_indices, primal_state_generation, primal_capacity, epoch, cells);
    const auto gates = gate_view(gate_ids, gate_ordinals, probabilities, admitted,
        gate_state_generation, epoch, cells);
    torch::Tensor output;
    auto packed = handle->ce_geo_projection();
    try {
        if (packed) {
            output = sparse_relation_forward_packed(handle, packed, input, edge_values,
                probabilities, admitted, primal_offsets, primal_ordinals, primal_ids,
                primal_gate_indices, gate_ids, gate_ordinals, primal_state_generation,
                primal_capacity, gate_state_generation, epoch, cells, stream);
        } else {
            auto weights_half = edge_values.to(torch::kFloat16).contiguous();
            output = torch::empty({input.size(0), handle->destination_count(), 64},
                torch::TensorOptions().device(input.device()).dtype(torch::kFloat32));
            const auto status = ss::relation_forward_cuda_v1(edge_view(handle), primal, gates,
                reinterpret_cast<const std::uint16_t*>(input.data_ptr<at::Half>()),
                handle->source_count(), handle->destination_count(),
                reinterpret_cast<const std::uint16_t*>(weights_half.data_ptr<at::Half>()),
                output.data_ptr<float>(), stream.stream());
            check_status(status, "sparse relation forward");
            c10::cuda::CUDACachingAllocator::recordStream(weights_half.storage().data_ptr(), stream);
        }
    } catch (...) {
        tape->retire();
        throw;
    }
    ctx->save_for_backward({input, edge_values, probabilities, admitted,
        primal_offsets, primal_ordinals, primal_ids, primal_gate_indices,
        derivative_offsets, derivative_ordinals, derivative_ids,
        derivative_gate_indices, gate_ids, gate_ordinals});
    ctx->saved_data["tape"] = c10::IValue::make_capsule(std::move(tape));
    ctx->saved_data["handle"] = c10::IValue::make_capsule(
        c10::make_intrusive<SparseHandleCapsule>(handle));
    ctx->saved_data["primal_state_generation"] = static_cast<std::int64_t>(primal_state_generation);
    ctx->saved_data["primal_capacity"] = static_cast<std::int64_t>(primal_capacity);
    ctx->saved_data["derivative_state_generation"] = static_cast<std::int64_t>(derivative_state_generation);
    ctx->saved_data["derivative_capacity"] = static_cast<std::int64_t>(derivative_capacity);
    ctx->saved_data["gate_state_generation"] = static_cast<std::int64_t>(gate_state_generation);
    c10::cuda::CUDACachingAllocator::recordStream(input.storage().data_ptr(), stream);
    c10::cuda::CUDACachingAllocator::recordStream(edge_values.storage().data_ptr(), stream);
    c10::cuda::CUDACachingAllocator::recordStream(output.storage().data_ptr(), stream);
    for (const auto* tensor : {&handle->edge_ids_device(),
            &handle->source_indices_device(), &handle->destination_indices_device()})
        c10::cuda::CUDACachingAllocator::recordStream(tensor->storage().data_ptr(), stream);
    for (const auto* tensor : {&probabilities, &admitted, &primal_offsets,
            &primal_ordinals, &primal_ids, &primal_gate_indices,
            &derivative_offsets, &derivative_ordinals, &derivative_ids,
            &derivative_gate_indices, &gate_ids, &gate_ordinals})
        c10::cuda::CUDACachingAllocator::recordStream(tensor->storage().data_ptr(), stream);
    return output;
}

torch::autograd::variable_list sparse_relation_backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
    TORCH_CHECK(grad_outputs.size() == 1 && grad_outputs.front().defined(),
                "sparse relation expects one output gradient");
    TORCH_CHECK(!torch::autograd::GradMode::is_enabled(),
                "sparse relation supports first-order gradients only");
    auto saved = ctx->get_saved_variables();
    auto input = saved[0], edge_values = saved[1], probabilities = saved[2], admitted = saved[3];
    auto primal_offsets = saved[4], primal_ordinals = saved[5], primal_ids = saved[6];
    auto primal_gate_indices = saved[7], derivative_offsets = saved[8];
    auto derivative_ordinals = saved[9], derivative_ids = saved[10];
    auto derivative_gate_indices = saved[11], gate_ids = saved[12], gate_ordinals = saved[13];
    auto capsule = c10::static_intrusive_pointer_cast<SparseHandleCapsule>(
        ctx->saved_data.at("handle").toCapsule());
    auto handle = capsule->handle;
    auto tape = c10::static_intrusive_pointer_cast<SparseTape>(
        ctx->saved_data.at("tape").toCapsule());
    bool expected = false;
    TORCH_CHECK(tape->consumed.compare_exchange_strong(expected, true),
                "sparse relation tape may be consumed only once");
    auto grad_output = grad_outputs.front().contiguous();
    TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32,
                "sparse relation output gradient must be FP32");
    auto grad_input = torch::zeros(input.sizes(),
        torch::TensorOptions().device(input.device()).dtype(torch::kFloat32));
    auto grad_weights = torch::zeros_like(edge_values);
    auto grad_gates = torch::zeros_like(probabilities);
    auto weights_half = edge_values.to(torch::kFloat16).contiguous();
    const auto cells = static_cast<std::uint32_t>(input.size(0));
    const auto epoch = handle->structure_epoch();
    auto primal = workset_view(primal_offsets, primal_ordinals, primal_ids,
        primal_gate_indices, ctx->saved_data.at("primal_state_generation").toInt(),
        ctx->saved_data.at("primal_capacity").toInt(), epoch, cells);
    auto derivative = workset_view(derivative_offsets, derivative_ordinals,
        derivative_ids, derivative_gate_indices,
        ctx->saved_data.at("derivative_state_generation").toInt(),
        ctx->saved_data.at("derivative_capacity").toInt(), epoch, cells);
    auto gates = gate_view(gate_ids, gate_ordinals, probabilities, admitted,
        ctx->saved_data.at("gate_state_generation").toInt(), epoch, cells);
    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    for (const auto* tensor : {&handle->edge_ids_device(),
            &handle->source_indices_device(), &handle->destination_indices_device()})
        c10::cuda::CUDACachingAllocator::recordStream(tensor->storage().data_ptr(), stream);
    auto status = ss::relation_backward_cuda_v1(edge_view(handle), primal, derivative,
        gates, reinterpret_cast<const std::uint16_t*>(input.data_ptr<at::Half>()),
        handle->source_count(), handle->destination_count(),
        reinterpret_cast<const std::uint16_t*>(weights_half.data_ptr<at::Half>()),
        grad_output.data_ptr<float>(), grad_input.data_ptr<float>(),
        grad_weights.data_ptr<float>(), grad_gates.data_ptr<float>(), stream.stream());
    check_status(status, "sparse relation backward");
    const auto retired = handle->support()->retire_forward_tape(epoch);
    TORCH_CHECK(retired == cellerator::compute::support::adaptive_support_status_v1::success,
                "could not retire sparse support structure tape");
    tape->retired.store(true, std::memory_order_release);
    c10::cuda::CUDACachingAllocator::recordStream(grad_output.storage().data_ptr(), stream);
    c10::cuda::CUDACachingAllocator::recordStream(grad_input.storage().data_ptr(), stream);
    c10::cuda::CUDACachingAllocator::recordStream(grad_weights.storage().data_ptr(), stream);
    c10::cuda::CUDACachingAllocator::recordStream(grad_gates.storage().data_ptr(), stream);
    torch::autograd::variable_list result(20);
    result[0] = grad_input.to(input.scalar_type());
    result[1] = grad_weights;
    result[2] = grad_gates;
    return result;
}

class SparseRelationAutograd final
    : public torch::autograd::Function<SparseRelationAutograd> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
            torch::Tensor input, torch::Tensor weights, torch::Tensor probabilities,
            torch::Tensor admitted, torch::Tensor primal_offsets,
            torch::Tensor primal_ordinals, torch::Tensor primal_ids,
            torch::Tensor primal_gate_indices, torch::Tensor derivative_offsets,
            torch::Tensor derivative_ordinals, torch::Tensor derivative_ids,
            torch::Tensor derivative_gate_indices, torch::Tensor gate_ids,
            torch::Tensor gate_ordinals, std::uint64_t primal_state_generation,
            std::uint64_t primal_capacity, std::uint64_t derivative_state_generation,
            std::uint64_t derivative_capacity, std::uint64_t gate_state_generation,
            SparseHandlePtr handle) {
        return sparse_relation_forward(ctx, std::move(input), std::move(weights),
            std::move(probabilities), std::move(admitted), std::move(primal_offsets),
            std::move(primal_ordinals), std::move(primal_ids),
            std::move(primal_gate_indices), std::move(derivative_offsets),
            std::move(derivative_ordinals), std::move(derivative_ids),
            std::move(derivative_gate_indices), std::move(gate_ids),
            std::move(gate_ordinals), primal_state_generation, primal_capacity,
            derivative_state_generation, derivative_capacity, gate_state_generation,
            std::move(handle));
    }
    static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::variable_list grad_outputs) {
        return sparse_relation_backward(ctx, std::move(grad_outputs));
    }
};

torch::Tensor sparse_latent_forward(torch::autograd::AutogradContext* ctx,
    torch::Tensor state, torch::Tensor input, torch::Tensor affected_offsets,
    torch::Tensor affected_mechanisms, torch::Tensor matrix_a,
    torch::Tensor matrix_b, torch::Tensor bias, SparseHandlePtr handle) {
    TORCH_CHECK(handle, "sparse support handle is null");
    const auto device = handle->device();
    validate_dense_cuda(state, "latent state", torch::kFloat16, device, 3);
    validate_dense_cuda(input, "latent input", torch::kFloat16, device, 2);
    validate_dense_cuda(affected_offsets, "affected offsets", torch::kInt64, device, 1);
    validate_dense_cuda(affected_mechanisms, "affected mechanisms", torch::kInt32, device, 1);
    validate_dense_cuda(matrix_a, "latent matrix A", torch::kFloat32, device, 3);
    validate_dense_cuda(matrix_b, "latent matrix B", torch::kFloat32, device, 3);
    validate_dense_cuda(bias, "latent bias", torch::kFloat32, device, 2);
    TORCH_CHECK(state.size(0) == input.size(0)
                && state.size(1) == matrix_a.size(0)
                && state.size(1) == matrix_b.size(0)
                && state.size(1) == bias.size(0)
                && state.size(2) == matrix_a.size(1)
                && state.size(2) == matrix_a.size(2)
                && input.size(1) == matrix_b.size(2)
                && matrix_b.size(1) == state.size(2)
                && bias.size(1) == state.size(2),
                "latent transition parameter shapes do not match state and input");
    TORCH_CHECK(affected_offsets.numel() == state.size(0) + 1,
                "affected offsets must have cells + 1 entries");
    const auto cells = static_cast<std::uint32_t>(state.size(0));
    const auto affected_count = static_cast<std::uint64_t>(affected_mechanisms.numel());
    ss::latent_transition_shape_v1 shape{cells,
        static_cast<std::uint32_t>(state.size(1)),
        static_cast<std::uint32_t>(state.size(2)),
        static_cast<std::uint32_t>(input.size(1))};
    auto a_half = matrix_a.to(torch::kFloat16).contiguous();
    auto b_half = matrix_b.to(torch::kFloat16).contiguous();
    auto bias_half = bias.to(torch::kFloat16).contiguous();
    auto output = state.to(torch::kFloat32).contiguous();
    auto stream = at::cuda::getCurrentCUDAStream(device);
    const auto status = ss::latent_transition_cuda_v1(shape,
        reinterpret_cast<const std::uint64_t*>(affected_offsets.data_ptr<std::int64_t>()),
        reinterpret_cast<const std::uint32_t*>(affected_mechanisms.data_ptr<std::int32_t>()),
        affected_count, reinterpret_cast<const std::uint16_t*>(state.data_ptr<at::Half>()),
        reinterpret_cast<const std::uint16_t*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const std::uint16_t*>(a_half.data_ptr<at::Half>()),
        reinterpret_cast<const std::uint16_t*>(b_half.data_ptr<at::Half>()),
        reinterpret_cast<const std::uint16_t*>(bias_half.data_ptr<at::Half>()),
        output.data_ptr<float>(), stream.stream());
    check_status(status, "sparse latent transition forward");
    ctx->save_for_backward({state, input, affected_offsets, affected_mechanisms,
                            matrix_a, matrix_b, bias, output});
    ctx->saved_data["handle"] = c10::IValue::make_capsule(
        c10::make_intrusive<SparseHandleCapsule>(handle));
    c10::cuda::CUDACachingAllocator::recordStream(state.storage().data_ptr(), stream);
    c10::cuda::CUDACachingAllocator::recordStream(input.storage().data_ptr(), stream);
    c10::cuda::CUDACachingAllocator::recordStream(output.storage().data_ptr(), stream);
    for (const auto* tensor : {&a_half, &b_half, &bias_half})
        c10::cuda::CUDACachingAllocator::recordStream(tensor->storage().data_ptr(), stream);
    return output;
}

torch::autograd::variable_list sparse_latent_backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
    TORCH_CHECK(grad_outputs.size() == 1 && grad_outputs.front().defined(),
                "sparse latent transition expects one output gradient");
    TORCH_CHECK(!torch::autograd::GradMode::is_enabled(),
                "sparse latent transition supports first-order gradients only");
    auto saved = ctx->get_saved_variables();
    auto state = saved[0], input = saved[1], offsets = saved[2], affected = saved[3];
    auto matrix_a = saved[4], matrix_b = saved[5], bias = saved[6], next_state = saved[7];
    auto handle_state = c10::static_intrusive_pointer_cast<SparseHandleCapsule>(
        ctx->saved_data.at("handle").toCapsule());
    auto handle = handle_state->handle;
    auto grad_next = grad_outputs.front().contiguous();
    TORCH_CHECK(grad_next.scalar_type() == torch::kFloat32,
                "latent state cotangent must be FP32");
    const auto cells = static_cast<std::uint32_t>(state.size(0));
    ss::latent_transition_shape_v1 shape{cells,
        static_cast<std::uint32_t>(state.size(1)),
        static_cast<std::uint32_t>(state.size(2)),
        static_cast<std::uint32_t>(input.size(1))};
    auto a_half = matrix_a.to(torch::kFloat16).contiguous();
    auto b_half = matrix_b.to(torch::kFloat16).contiguous();
    auto grad_state = grad_next.contiguous().clone();
    auto grad_input = torch::zeros_like(input, torch::TensorOptions().dtype(torch::kFloat32));
    auto grad_a = torch::zeros_like(matrix_a);
    auto grad_b = torch::zeros_like(matrix_b);
    auto grad_bias = torch::zeros_like(bias);
    auto stream = at::cuda::getCurrentCUDAStream(state.get_device());
    const auto status = ss::latent_transition_vjp_cuda_v1(shape,
        reinterpret_cast<const std::uint64_t*>(offsets.data_ptr<std::int64_t>()),
        reinterpret_cast<const std::uint32_t*>(affected.data_ptr<std::int32_t>()),
        static_cast<std::uint64_t>(affected.numel()),
        reinterpret_cast<const std::uint16_t*>(state.data_ptr<at::Half>()),
        reinterpret_cast<const std::uint16_t*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const std::uint16_t*>(a_half.data_ptr<at::Half>()),
        reinterpret_cast<const std::uint16_t*>(b_half.data_ptr<at::Half>()),
        reinterpret_cast<const float*>(next_state.data_ptr<float>()),
        grad_next.data_ptr<float>(), grad_state.data_ptr<float>(),
        grad_input.data_ptr<float>(), grad_a.data_ptr<float>(),
        grad_b.data_ptr<float>(), grad_bias.data_ptr<float>(), stream.stream());
    check_status(status, "sparse latent transition backward");
    for (const auto* tensor : {&grad_next, &grad_state, &grad_input, &grad_a, &grad_b, &grad_bias})
        c10::cuda::CUDACachingAllocator::recordStream(tensor->storage().data_ptr(), stream);
    torch::autograd::variable_list result(8);
    result[0] = grad_state.to(state.scalar_type());
    result[1] = grad_input.to(input.scalar_type());
    result[4] = grad_a;
    result[5] = grad_b;
    result[6] = grad_bias;
    return result;
}

class SparseLatentAutograd final
    : public torch::autograd::Function<SparseLatentAutograd> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
            torch::Tensor state, torch::Tensor input, torch::Tensor offsets,
            torch::Tensor affected, torch::Tensor matrix_a, torch::Tensor matrix_b,
            torch::Tensor bias, SparseHandlePtr handle) {
        return sparse_latent_forward(ctx, std::move(state), std::move(input),
            std::move(offsets), std::move(affected), std::move(matrix_a),
            std::move(matrix_b), std::move(bias), std::move(handle));
    }
    static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::variable_list grad_outputs) {
        return sparse_latent_backward(ctx, std::move(grad_outputs));
    }
};

SparseHandlePtr sparse_create_handle(
        const std::vector<std::uint64_t>& edge_ids,
        const std::vector<std::uint32_t>& sources,
        const std::vector<std::uint32_t>& destinations,
        std::uint32_t source_count, std::uint32_t destination_count,
        std::uint64_t structure_epoch, const std::vector<std::uint64_t>& axis_words,
        const std::vector<std::pair<std::uint64_t, std::uint64_t>>& block_ids,
        torch::Tensor initial, const py::list& block_values, int device,
        std::uint32_t reader_capacity,
        const std::shared_ptr<sb::native_support_handle_v1>& support) {
    TORCH_CHECK(support, "native support handle is required");
    TORCH_CHECK(edge_ids.size() == sources.size() && edge_ids.size() == destinations.size(),
                "edge topology arrays must have matching lengths");
    TORCH_CHECK(std::is_sorted(edge_ids.begin(), edge_ids.end())
                && std::adjacent_find(edge_ids.begin(), edge_ids.end()) == edge_ids.end(),
                "logical edge IDs must be unique and sorted");
    TORCH_CHECK(axis_words.size() == 8, "parameter axis requires eight identity words");
    TORCH_CHECK(initial.defined() && initial.device().is_cpu()
                && initial.scalar_type() == torch::kFloat32 && initial.dim() == 1
                && initial.is_contiguous(),
                "initial parameters must be contiguous host FP32");
    TORCH_CHECK(block_ids.size() == static_cast<std::size_t>(initial.numel()),
                "one stable parameter-block ID is required per flat scalar");
    for (std::size_t i = 0; i < sources.size(); ++i)
        TORCH_CHECK(sources[i] < source_count && destinations[i] < destination_count,
                    "edge endpoint ordinal is out of range");
    auto support_snapshot = support->snapshot_copy();
    TORCH_CHECK(support_snapshot.structure_epoch == structure_epoch,
                "support manager and relation handle structure epochs differ");
    std::vector<std::pair<std::uint64_t, std::pair<std::uint32_t, std::uint32_t>>>
        snapshot_edges;
    auto append_snapshot_edges = [&](const cellerator::compute::support::logical_support_view_v1* view) {
        TORCH_CHECK(view != nullptr, "support snapshot is incomplete");
        for (std::uint64_t i = 0; i < view->edge_count; ++i) {
            const auto& edge = view->edges[i];
            TORCH_CHECK(edge.participant_count == 2,
                        "pair relation execution requires exactly two ordered participants per edge");
            const auto& source = view->participants[edge.participant_begin];
            const auto& destination = view->participants[edge.participant_begin + 1];
            TORCH_CHECK(source.role == 0 && destination.role == 1
                        && source.entity_id > 0 && destination.entity_id > 0
                        && source.entity_id - 1 <= std::numeric_limits<std::uint32_t>::max()
                        && destination.entity_id - 1 <= std::numeric_limits<std::uint32_t>::max(),
                        "pair relation endpoints must use roles 0/1 and uint32 entity ordinals");
            snapshot_edges.push_back({edge.logical_edge_id,
                {static_cast<std::uint32_t>(source.entity_id - 1),
                 static_cast<std::uint32_t>(destination.entity_id - 1)}});
        }
    };
    const auto base_view = support_snapshot.base.view();
    const auto delta_view = support_snapshot.delta.view();
    append_snapshot_edges(&base_view);
    append_snapshot_edges(&delta_view);
    std::sort(snapshot_edges.begin(), snapshot_edges.end(),
        [](const auto& left, const auto& right) { return left.first < right.first; });
    TORCH_CHECK(snapshot_edges.size() == edge_ids.size(),
                "relation edge arrays do not cover the active logical support snapshot");
    for (std::size_t i = 0; i < edge_ids.size(); ++i)
        TORCH_CHECK(snapshot_edges[i].first == edge_ids[i]
                    && snapshot_edges[i].second.first == sources[i]
                    && snapshot_edges[i].second.second == destinations[i],
                    "relation endpoints or edge IDs differ from the active logical support snapshot");

    auto u64 = [](std::uint64_t value) { return value; };
    cellerator::execution::persistent_axis_identity axis{};
    axis.header.schema_version = cellerator::execution::biological_abi_version;
    axis.header.kind = cellerator::execution::serialized_record_kind::persistent_axis_identity;
    axis.header.byte_count = sizeof(cellerator::execution::persistent_axis_identity);
    axis.domain = {u64(axis_words[0]), u64(axis_words[1])};
    axis.order = {u64(axis_words[2]), u64(axis_words[3])};
    axis.geometry = {u64(axis_words[4]), u64(axis_words[5])};
    axis.partition = {u64(axis_words[6]), u64(axis_words[7])};
    ix::indexed_axis parameter_axis{axis, static_cast<std::uint64_t>(initial.numel())};

    std::vector<ix::identity> ids;
    ids.reserve(block_ids.size());
    for (const auto& [low, high] : block_ids) ids.push_back({low, high});
    std::vector<ss::parameter_block_v1> blocks;
    blocks.reserve(block_values.size());
    for (const auto item : block_values) {
        py::dict value = py::cast<py::dict>(item);
        ss::parameter_block_v1 block;
        block.name = py::cast<std::string>(value["name"]);
        block.offset = py::cast<std::uint64_t>(value["offset"]);
        block.shape = py::cast<std::vector<std::uint64_t>>(value["shape"]);
        block.numel = py::cast<std::uint64_t>(value["numel"]);
        blocks.push_back(std::move(block));
    }
    auto initial_cpu = initial.contiguous();
    auto parameters = std::make_shared<ss::sparse_parameter_owner_v1>(
        parameter_axis, std::move(ids),
        std::span<const float>(initial_cpu.data_ptr<float>(), initial_cpu.numel()),
        std::move(blocks), device, reader_capacity);
    auto handle = std::make_shared<SparseHandle>(support, std::move(parameters),
        edge_ids, sources, destinations, source_count, destination_count,
        structure_epoch, device);
    handle->set_endpoint_tensors(
        copy_vector_to_device(edge_ids, torch::kUInt64, device),
        copy_vector_to_device(sources, torch::kInt32, device),
        copy_vector_to_device(destinations, torch::kInt32, device));
    auto creation_stream = at::cuda::getCurrentCUDAStream(device);
    handle->record_endpoints_ready(creation_stream.stream());
    return handle;
}

SparseHandlePtr sparse_rebind_handle(
        const SparseHandlePtr& previous,
        const std::vector<std::uint64_t>& edge_ids,
        const std::vector<std::uint32_t>& sources,
        const std::vector<std::uint32_t>& destinations,
        std::uint64_t structure_epoch) {
    TORCH_CHECK(previous, "sparse support handle is null");
    TORCH_CHECK(edge_ids.size() == sources.size()
                && edge_ids.size() == destinations.size()
                && std::is_sorted(edge_ids.begin(), edge_ids.end())
                && std::adjacent_find(edge_ids.begin(), edge_ids.end()) == edge_ids.end(),
                "replacement pair support arrays must be sorted and have matching lengths");
    auto snapshot = previous->support()->snapshot_copy();
    TORCH_CHECK(snapshot.structure_epoch == structure_epoch,
                "replacement sparse support epoch is not published");
    std::vector<std::pair<std::uint64_t, std::pair<std::uint32_t, std::uint32_t>>> expected;
    auto append = [&](const cellerator::compute::support::logical_support_view_v1* view) {
        TORCH_CHECK(view != nullptr, "replacement support snapshot is incomplete");
        for (std::uint64_t i = 0; i < view->edge_count; ++i) {
            const auto& edge = view->edges[i];
            TORCH_CHECK(edge.participant_count == 2,
                        "numeric relation projection requires exactly two participants per edge");
            const auto& src = view->participants[edge.participant_begin];
            const auto& dst = view->participants[edge.participant_begin + 1];
            TORCH_CHECK(src.role == 0 && dst.role == 1 && src.entity_id > 0
                        && dst.entity_id > 0
                        && src.entity_id - 1 <= std::numeric_limits<std::uint32_t>::max()
                        && dst.entity_id - 1 <= std::numeric_limits<std::uint32_t>::max(),
                        "replacement support endpoints must have ordered roles 0/1");
            expected.push_back({edge.logical_edge_id,
                {static_cast<std::uint32_t>(src.entity_id - 1),
                 static_cast<std::uint32_t>(dst.entity_id - 1)}});
        }
    };
    const auto base_view = snapshot.base.view();
    const auto delta_view = snapshot.delta.view();
    append(&base_view);
    append(&delta_view);
    std::sort(expected.begin(), expected.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });
    TORCH_CHECK(expected.size() == edge_ids.size(),
                "replacement arrays do not cover active logical support");
    for (std::size_t i = 0; i < edge_ids.size(); ++i)
        TORCH_CHECK(expected[i].first == edge_ids[i]
                    && expected[i].second.first == sources[i]
                    && expected[i].second.second == destinations[i]
                    && sources[i] < previous->source_count()
                    && destinations[i] < previous->destination_count(),
                    "replacement arrays differ from published logical support");
    auto rebound = std::make_shared<SparseHandle>(previous->support(), previous->parameters(),
        edge_ids, sources, destinations, previous->source_count(),
        previous->destination_count(), structure_epoch, previous->device());
    rebound->set_endpoint_tensors(
        copy_vector_to_device(edge_ids, torch::kUInt64, previous->device()),
        copy_vector_to_device(sources, torch::kInt32, previous->device()),
        copy_vector_to_device(destinations, torch::kInt32, previous->device()));
    auto binding_stream = at::cuda::getCurrentCUDAStream(previous->device());
    rebound->record_endpoints_ready(binding_stream.stream());
    auto prior_projection = previous->ce_geo_projection();
    if (prior_projection) {
        TORCH_CHECK(prior_projection->projection_edge_ids_host.size()
                        == previous->edge_ids().size(),
                    "prepared projection map no longer matches logical support order");
        std::unordered_map<std::uint64_t, std::uint64_t> projection_ids_by_logical_id;
        for (std::size_t index = 0; index < previous->edge_ids().size(); ++index)
            projection_ids_by_logical_id.emplace(previous->edge_ids()[index],
                prior_projection->projection_edge_ids_host[index]);
        std::vector<std::uint64_t> new_projection_edge_ids;
        new_projection_edge_ids.reserve(edge_ids.size());
        for (const auto edge_id : edge_ids) {
            const auto found = projection_ids_by_logical_id.find(edge_id);
            new_projection_edge_ids.push_back(found == projection_ids_by_logical_id.end()
                ? std::numeric_limits<std::uint64_t>::max() : found->second);
        }
        ss::ce_geo_n64_projection_binding_v1 binding{};
        check_status(ss::prepare_ce_geo_n64_projection_v1(
            prior_projection->image_bytes.data(), prior_projection->image_bytes.size(),
            new_projection_edge_ids, sources, destinations,
            prior_projection->source_group_members_host,
            prior_projection->destination_group_members_host, previous->source_count(),
            previous->destination_count(), structure_epoch, prior_projection->projection_id,
            &binding), "CE-GEO support revision projection preparation");
        auto replacement = std::make_shared<sb::prepared_ce_geo_n64_torch_v1>();
        replacement->binding = std::move(binding);
        replacement->image_bytes = prior_projection->image_bytes;
        replacement->source_group_members_host = prior_projection->source_group_members_host;
        replacement->destination_group_members_host = prior_projection->destination_group_members_host;
        replacement->max_cells = prior_projection->max_cells;
        replacement->max_primal_items = prior_projection->max_primal_items;
        replacement->projection_id = prior_projection->projection_id;
        replacement->projection_edge_ids_host = new_projection_edge_ids;
        auto host_image = torch::empty({static_cast<std::int64_t>(replacement->image_bytes.size())},
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
        std::memcpy(host_image.data_ptr<std::uint8_t>(), replacement->image_bytes.data(),
                    replacement->image_bytes.size());
        replacement->image = host_image.to(torch::Device(torch::kCUDA, previous->device()));
        replacement->edge_value_map_indices = copy_vector_to_device(
            replacement->binding.edge_value_map_indices, torch::kUInt64, previous->device());
        replacement->projection_edge_ids = copy_vector_to_device(
            replacement->binding.projection_edge_ids, torch::kUInt64, previous->device());
        replacement->tile_destination_groups = copy_vector_to_device(
            replacement->binding.tile_destination_groups, torch::kInt32, previous->device());
        replacement->tile_source_bases = copy_vector_to_device(
            replacement->binding.tile_source_bases, torch::kInt32, previous->device());
        replacement->source_members = copy_vector_to_device(
            replacement->binding.source_members, torch::kInt32, previous->device());
        replacement->destination_members = copy_vector_to_device(
            replacement->binding.destination_members, torch::kInt32, previous->device());
        replacement->logical_source_to_physical = copy_vector_to_device(
            replacement->binding.logical_source_to_physical, torch::kInt32, previous->device());
        auto stream = binding_stream;
        replacement->record_ready(stream.stream());
        rebound->set_ce_geo_projection(std::move(replacement));
        (void)ce_geo_work_buffers(rebound->ce_geo_projection(),
            prior_projection->max_cells, stream);
    }
    return rebound;
}

py::dict sparse_prepare_ce_geo_n64(
        const SparseHandlePtr& handle, py::bytes image_blob,
        const std::vector<std::uint32_t>& source_group_members,
        const std::vector<std::uint32_t>& destination_group_members,
        std::uint64_t projection_id, std::uint32_t max_cells,
        std::uint64_t max_primal_items,
        std::vector<std::uint64_t> geometry_edge_ids) {
    TORCH_CHECK(handle, "sparse support handle is null");
    TORCH_CHECK(max_cells > 0 && max_primal_items > 0,
                "CE-GEO preparation capacities must be positive");
    char* image_data = nullptr;
    Py_ssize_t image_size = 0;
    if (PyBytes_AsStringAndSize(image_blob.ptr(), &image_data, &image_size) != 0)
        throw py::error_already_set();
    TORCH_CHECK(image_size > 0, "CE-GEO image must not be empty");
    if (geometry_edge_ids.empty()) geometry_edge_ids = handle->edge_ids();
    TORCH_CHECK(geometry_edge_ids.size() == handle->edge_ids().size(),
                "geometry_edge_ids must align with stable logical support order");
    std::unordered_map<std::uint64_t, std::uint8_t> seen_projection_ids;
    for (const auto projection_edge_id : geometry_edge_ids) {
        if (projection_edge_id == std::numeric_limits<std::uint64_t>::max()) continue;
        TORCH_CHECK(seen_projection_ids.emplace(projection_edge_id, 1u).second,
                    "mapped geometry edge IDs must be unique in logical support order");
    }
    ss::ce_geo_n64_projection_binding_v1 binding{};
    const auto status = ss::prepare_ce_geo_n64_projection_v1(
        image_data, static_cast<std::size_t>(image_size), geometry_edge_ids,
        handle->sources(), handle->destinations(), source_group_members,
        destination_group_members, handle->source_count(),
        handle->destination_count(), handle->structure_epoch(), projection_id,
        &binding);
    check_status(status, "CE-GEO N64 projection preparation");

    auto prepared = std::make_shared<sb::prepared_ce_geo_n64_torch_v1>();
    prepared->binding = std::move(binding);
    prepared->projection_edge_ids_host = geometry_edge_ids;
    prepared->image_bytes.assign(reinterpret_cast<std::uint8_t*>(image_data),
        reinterpret_cast<std::uint8_t*>(image_data) + image_size);
    prepared->source_group_members_host = source_group_members;
    prepared->destination_group_members_host = destination_group_members;
    prepared->max_cells = max_cells;
    prepared->max_primal_items = max_primal_items;
    prepared->projection_id = projection_id;
    auto host_image = torch::empty({image_size},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    std::memcpy(host_image.data_ptr<std::uint8_t>(), image_data,
                static_cast<std::size_t>(image_size));
    prepared->image = host_image.to(torch::Device(torch::kCUDA, handle->device()));
    prepared->edge_value_map_indices = copy_vector_to_device(
        prepared->binding.edge_value_map_indices, torch::kUInt64, handle->device());
    prepared->projection_edge_ids = copy_vector_to_device(
        prepared->binding.projection_edge_ids, torch::kUInt64, handle->device());
    prepared->tile_destination_groups = copy_vector_to_device(
        prepared->binding.tile_destination_groups, torch::kInt32, handle->device());
    prepared->tile_source_bases = copy_vector_to_device(
        prepared->binding.tile_source_bases, torch::kInt32, handle->device());
    prepared->source_members = copy_vector_to_device(
        prepared->binding.source_members, torch::kInt32, handle->device());
    prepared->destination_members = copy_vector_to_device(
        prepared->binding.destination_members, torch::kInt32, handle->device());
    prepared->logical_source_to_physical = copy_vector_to_device(
        prepared->binding.logical_source_to_physical, torch::kInt32, handle->device());
    auto stream = at::cuda::getCurrentCUDAStream(handle->device());
    prepared->record_ready(stream.stream());
    handle->set_ce_geo_projection(prepared);
    (void)ce_geo_work_buffers(prepared, max_cells, stream);

    py::dict result;
    result["structure_epoch"] = prepared->binding.structure_epoch;
    result["projection_id"] = prepared->binding.projection_id;
    result["tile_count"] = prepared->binding.header.tile_count;
    result["value_map_count"] = prepared->binding.header.value_map_count;
    result["physical_source_count"] = prepared->binding.source_members.size();
    result["destination_group_count"] = prepared->binding.header.destination_group_count;
    return result;
}

void sparse_reserve_ce_geo_workspace(const SparseHandlePtr& handle) {
    auto packed = handle->ce_geo_projection();
    TORCH_CHECK(packed, "CE-GEO N64 projection has not been prepared");
    auto stream = at::cuda::getCurrentCUDAStream(handle->device());
    (void)ce_geo_work_buffers(packed, packed->max_cells, stream);
}

py::tuple sparse_endpoint_indices(const SparseHandlePtr& handle) {
    TORCH_CHECK(handle, "sparse support handle is null");
    return py::make_tuple(handle->source_indices_device(),
                          handle->destination_indices_device());
}

torch::Tensor sparse_parameters(const SparseHandlePtr& handle) {
    auto state = alias_state(handle);
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->tensor.defined()) return state->tensor;
    auto owner = handle->parameters()->native_owner();
    auto options = torch::TensorOptions().dtype(torch::kFloat32)
        .device(torch::Device(torch::kCUDA, owner->device())).requires_grad(false);
    {
        std::lock_guard<std::mutex> registry_lock(registry_mutex);
        owner_registry[owner->data()] = owner;
    }
    state->tensor = torch::from_blob(owner->data(),
        {static_cast<std::int64_t>(owner->size())}, [handle](void*) {}, options)
        .set_requires_grad(true);
    state->aliases.emplace_back(state->tensor, tensor_version(state->tensor));
    return state->tensor;
}

void sparse_validate_parameters(const SparseHandlePtr& handle, const torch::Tensor& tensor) {
    auto state = alias_state(handle);
    std::lock_guard<std::mutex> lock(state->mutex);
    validate_parameters_locked(tensor, state, handle);
}

void sparse_preflight_write(const SparseHandlePtr& handle) {
    handle->parameters()->native_owner()->preflight_write();
}

void sparse_begin_update(const SparseHandlePtr& handle) {
    auto state = alias_state(handle);
    std::lock_guard<std::mutex> lock(state->mutex);
    TORCH_CHECK(state->tensor.defined(), "sparse parameter tensor has not been attached");
    validate_parameters_locked(state->tensor, state, handle);
    auto stream = at::cuda::getCurrentCUDAStream(handle->device());
    handle->parameters()->native_owner()->begin_write(stream.stream());
}

void sparse_publish_update(const SparseHandlePtr& handle) {
    auto state = alias_state(handle);
    std::lock_guard<std::mutex> lock(state->mutex);
    auto stream = at::cuda::getCurrentCUDAStream(handle->device());
    handle->parameters()->native_owner()->publish_write(stream.stream());
    for (auto& alias : state->aliases) alias.second = tensor_version(alias.first);
}

void sparse_poison(const SparseHandlePtr& handle) {
    handle->parameters()->native_owner()->poison();
}

py::tuple sparse_snapshot(const SparseHandlePtr& handle) {
    auto stream = at::cuda::getCurrentCUDAStream(handle->device());
    const auto values = handle->parameters()->native_owner()->snapshot(stream.stream());
    auto tensor = torch::empty({static_cast<std::int64_t>(values.size())},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    std::copy(values.begin(), values.end(), tensor.data_ptr<float>());
    py::list metadata;
    for (const auto& block : handle->parameters()->blocks()) {
        py::dict value;
        value["name"] = block.name;
        value["offset"] = block.offset;
        value["shape"] = block.shape;
        value["numel"] = block.numel;
        metadata.append(value);
    }
    return py::make_tuple(std::move(tensor), std::move(metadata));
}

void sparse_restore(const SparseHandlePtr& handle, const torch::Tensor& values,
                    const py::list& metadata) {
    TORCH_CHECK(values.device().is_cpu() && values.scalar_type() == torch::kFloat32
                && values.dim() == 1 && values.is_contiguous(),
                "restore values must be contiguous CPU FP32");
    TORCH_CHECK(metadata.size() == handle->parameters()->blocks().size(),
                "restore parameter block count differs from prepared owner");
    const auto& blocks = handle->parameters()->blocks();
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        const auto item = py::cast<py::dict>(metadata[i]);
        TORCH_CHECK(py::cast<std::string>(item["name"]) == blocks[i].name
                    && py::cast<std::uint64_t>(item["offset"]) == blocks[i].offset
                    && py::cast<std::uint64_t>(item["numel"]) == blocks[i].numel
                    && py::cast<std::vector<std::uint64_t>>(item["shape"]) == blocks[i].shape,
                    "restore parameter block metadata mismatch");
    }
    TORCH_CHECK(values.numel() == static_cast<std::int64_t>(handle->parameters()->parameter_count()),
                "restore parameter count mismatch");
    auto stream = at::cuda::getCurrentCUDAStream(handle->device());
    handle->parameters()->native_owner()->restore(
        std::span<const float>(values.data_ptr<float>(), values.numel()), stream.stream());
    auto state = alias_state(handle);
    std::lock_guard<std::mutex> lock(state->mutex);
    for (auto& alias : state->aliases) alias.second = tensor_version(alias.first);
}

py::list sparse_parameter_blocks(const SparseHandlePtr& handle) {
    py::list result;
    for (const auto& block : handle->parameters()->blocks()) {
        py::dict value;
        value["name"] = py::cast(block.name);
        value["offset"] = py::cast(block.offset);
        value["shape"] = py::cast(block.shape);
        value["numel"] = py::cast(block.numel);
        result.append(std::move(value));
    }
    return result;
}

std::uint32_t sparse_pin_read(const SparseHandlePtr& handle) {
    auto stream = at::cuda::getCurrentCUDAStream(handle->device());
    return handle->parameters()->native_owner()->acquire_reader(stream.stream());
}

void sparse_complete_read(const SparseHandlePtr& handle, std::uint32_t ticket) {
    auto stream = at::cuda::getCurrentCUDAStream(handle->device());
    handle->parameters()->native_owner()->complete_reader(ticket, stream.stream());
}

bool is_sparse_native_owner(const torch::Tensor& tensor,
                            const std::vector<SparseHandlePtr>& handles) {
    if (!tensor.defined() || tensor.numel() == 0) return false;
    const auto address = reinterpret_cast<std::uintptr_t>(tensor.data_ptr());
    const auto bytes = static_cast<std::size_t>(tensor.nbytes());
    for (const auto& handle : handles) {
        if (!handle) continue;
        const auto& owner = handle->parameters()->native_owner();
        const auto begin = reinterpret_cast<std::uintptr_t>(owner->data());
        const auto end = begin + owner->size() * sizeof(float);
        if (address < end && begin < address + bytes) return true;
    }
    std::lock_guard<std::mutex> lock(registry_mutex);
    for (auto it = owner_registry.begin(); it != owner_registry.end();) {
        auto owner = it->second.lock();
        if (!owner) { it = owner_registry.erase(it); continue; }
        const auto begin = reinterpret_cast<std::uintptr_t>(owner->data());
        const auto end = begin + owner->size() * sizeof(float);
        if (address < end && begin < address + bytes) return true;
        ++it;
    }
    return false;
}

torch::Tensor sparse_relation_apply(
        const torch::Tensor& input, const torch::Tensor& weights,
        const torch::Tensor& probabilities, const torch::Tensor& admitted,
        const torch::Tensor& primal_offsets, const torch::Tensor& primal_ordinals,
        const torch::Tensor& primal_ids, const torch::Tensor& primal_gate_indices,
        const torch::Tensor& derivative_offsets, const torch::Tensor& derivative_ordinals,
        const torch::Tensor& derivative_ids, const torch::Tensor& derivative_gate_indices,
        const torch::Tensor& gate_ids, const torch::Tensor& gate_ordinals,
        std::uint64_t primal_state_generation, std::uint64_t primal_capacity,
        std::uint64_t derivative_state_generation, std::uint64_t derivative_capacity,
        std::uint64_t gate_state_generation, const SparseHandlePtr& handle) {
    return SparseRelationAutograd::apply(input, weights, probabilities, admitted,
        primal_offsets, primal_ordinals, primal_ids, primal_gate_indices,
        derivative_offsets, derivative_ordinals, derivative_ids, derivative_gate_indices,
        gate_ids, gate_ordinals, primal_state_generation, primal_capacity,
        derivative_state_generation, derivative_capacity, gate_state_generation, handle);
}

torch::Tensor sparse_latent_transition(
        const torch::Tensor& state, const torch::Tensor& input,
        const torch::Tensor& affected_offsets, const torch::Tensor& affected_mechanisms,
        const torch::Tensor& matrix_a, const torch::Tensor& matrix_b,
        const torch::Tensor& bias, const SparseHandlePtr& handle) {
    return SparseLatentAutograd::apply(state, input, affected_offsets,
        affected_mechanisms, matrix_a, matrix_b, bias, handle);
}

} // namespace

void bind_sparse_support(py::module_& module) {
    ::bind_sparse_support_indexing(module);
    ::bind_sparse_support_geometry(module);
    py::class_<SparseHandle, SparseHandlePtr>(module, "SparseTorchHandle")
        .def_property_readonly("support", &SparseHandle::support);
    module.def("sparse_create_handle", &sparse_create_handle,
        py::arg("edge_ids"), py::arg("sources"), py::arg("destinations"),
        py::arg("source_count"), py::arg("destination_count"),
        py::arg("structure_epoch"), py::arg("parameter_axis_words"),
        py::arg("block_ids"), py::arg("initial"), py::arg("blocks"),
        py::arg("device"), py::arg("reader_capacity"), py::arg("support"));
    module.def("sparse_rebind_handle", &sparse_rebind_handle,
        py::arg("previous"), py::arg("edge_ids"), py::arg("sources"),
        py::arg("destinations"), py::arg("structure_epoch"));
    module.def("sparse_prepare_ce_geo_n64", &sparse_prepare_ce_geo_n64,
        py::arg("handle"), py::arg("image"), py::arg("source_group_members"),
        py::arg("destination_group_members"), py::arg("projection_id"),
        py::arg("max_cells"), py::arg("max_primal_items"),
        py::arg("geometry_edge_ids") = std::vector<std::uint64_t>{});
    module.def("sparse_reserve_ce_geo_workspace", &sparse_reserve_ce_geo_workspace,
        py::arg("handle"));
    module.def("sparse_endpoint_indices", &sparse_endpoint_indices,
        py::arg("handle"));
    module.def("sparse_parameters", &sparse_parameters);
    module.def("sparse_validate_parameters", &sparse_validate_parameters);
    module.def("sparse_generation", [](const SparseHandlePtr& handle) {
        return handle->parameters()->native_owner()->generation();
    });
    module.def("sparse_preflight_write", &sparse_preflight_write);
    module.def("sparse_begin_update", &sparse_begin_update);
    module.def("sparse_publish_update", &sparse_publish_update);
    module.def("sparse_poison", &sparse_poison);
    module.def("sparse_snapshot", &sparse_snapshot);
    module.def("sparse_restore", &sparse_restore);
    module.def("sparse_parameter_blocks", &sparse_parameter_blocks);
    module.def("sparse_structure_epoch", [](const SparseHandlePtr& handle) {
        return handle->structure_epoch();
    });
    module.def("sparse_pin_read", &sparse_pin_read);
    module.def("sparse_complete_read", &sparse_complete_read);
    module.def("is_sparse_native_owner", &is_sparse_native_owner);
    module.def("sparse_relation_apply", &sparse_relation_apply,
        py::arg("input"), py::arg("weights"), py::arg("probabilities"),
        py::arg("admitted"), py::arg("primal_offsets"), py::arg("primal_ordinals"),
        py::arg("primal_ids"), py::arg("primal_gate_indices"),
        py::arg("derivative_offsets"), py::arg("derivative_ordinals"),
        py::arg("derivative_ids"), py::arg("derivative_gate_indices"),
        py::arg("gate_ids"), py::arg("gate_ordinals"),
        py::arg("primal_state_generation"), py::arg("primal_capacity"),
        py::arg("derivative_state_generation"), py::arg("derivative_capacity"),
        py::arg("gate_state_generation"), py::arg("handle"));
    module.def("sparse_latent_transition", &sparse_latent_transition,
        py::arg("state"), py::arg("input"), py::arg("affected_offsets"),
        py::arg("affected_mechanisms"), py::arg("matrix_a"), py::arg("matrix_b"),
        py::arg("bias"), py::arg("handle"));
}
