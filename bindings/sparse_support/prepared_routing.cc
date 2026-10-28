#include "prepared_routing.hh"

#include <Cellerator/ops/sparse_support/prepared_routing.hh>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;
namespace ce_indexing = cellerator::compute::indexing;
namespace ce_routing = cellerator::ops::sparse_support;

namespace {

static_assert(sizeof(ce_indexing::posting_term_v1) == 16u,
              "posting terms must remain a compact two-word CUDA record");
static_assert(sizeof(ce_indexing::indexed_work_item_v1) == 16u,
              "posting outputs must remain a compact two-word CUDA record");
static_assert(sizeof(ce_indexing::cuda_posting_scratch_v1::cursor_v1) == 32u,
              "posting query cursors must remain four int64 words");
static_assert(sizeof(ce_indexing::cuda_posting_status_v1) == 16u,
              "posting query status must remain two int64 words");

using TorchTensor = torch::Tensor;

struct QueryWorkspace {
    std::mutex use_mutex;
    std::uint64_t query_capacity = 0u;
    std::uint64_t cell_capacity = 0u;
    std::uint64_t output_capacity = 0u;
    TorchTensor queries;       // int64 [query_capacity, 2], posting_term_v1 view
    TorchTensor query_offsets; // int64 [cell_capacity + 1]
    TorchTensor output_items;  // int64 [output_capacity, 2], indexed_work_item_v1
    TorchTensor cursors;       // int64 [query_capacity, 4]
    TorchTensor counts;        // int64 [cell_capacity]
    TorchTensor device_status; // int64 [2], cuda_posting_status_v1
    TorchTensor host_status;   // pinned CPU int64 [2]
};

void record_stream(const TorchTensor &tensor, c10::cuda::CUDAStream stream) {
    if (tensor.defined() && tensor.is_cuda() && tensor.numel() != 0)
        c10::cuda::CUDACachingAllocator::recordStream(
            tensor.storage().data_ptr(), stream);
}

TorchTensor allocate_cuda(std::vector<std::int64_t> shape, at::ScalarType type,
                          std::int64_t device) {
    return at::empty(shape, at::TensorOptions().dtype(type)
        .device(at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(device))));
}

template <typename T>
TorchTensor upload_vector(const std::vector<T> &source,
                          std::int64_t rows, std::int64_t columns,
                          std::int64_t device) {
    TORCH_CHECK(rows >= 0 && columns >= 0
                && static_cast<std::uint64_t>(rows)
                    <= std::numeric_limits<std::uint64_t>::max()
                        / std::max<std::uint64_t>(1u,
                            static_cast<std::uint64_t>(columns)),
                "prepared routing projection shape overflows");
    const auto count = static_cast<std::uint64_t>(rows)
        * static_cast<std::uint64_t>(columns);
    TORCH_CHECK(count <= std::numeric_limits<std::size_t>::max()
                    / sizeof(std::int64_t)
                && source.size() <= std::numeric_limits<std::size_t>::max()
                    / sizeof(T)
                && count * sizeof(std::int64_t) == source.size() * sizeof(T),
                "prepared routing upload shape mismatch");
    const auto cpu = at::empty({rows, columns}, at::TensorOptions()
        .dtype(at::kLong).device(at::kCPU));
    if (!source.empty())
        std::memcpy(cpu.data_ptr(), source.data(), source.size() * sizeof(T));
    auto gpu = allocate_cuda({rows, columns}, at::kLong, device);
    if (count != 0u) gpu.copy_(cpu, false);
    return gpu;
}

TorchTensor upload_offsets(const std::vector<std::uint64_t> &source,
                           std::int64_t device) {
    TORCH_CHECK(source.size() <= static_cast<std::size_t>(
                    std::numeric_limits<std::int64_t>::max()),
                "prepared routing offset projection is too large");
    const auto cpu = at::empty({static_cast<std::int64_t>(source.size())},
        at::TensorOptions().dtype(at::kLong).device(at::kCPU));
    if (!source.empty())
        std::memcpy(cpu.data_ptr(), source.data(), source.size() * sizeof(std::uint64_t));
    auto gpu = allocate_cuda({static_cast<std::int64_t>(source.size())},
                             at::kLong, device);
    if (!source.empty()) gpu.copy_(cpu, false);
    return gpu;
}

class TorchPreparedEntityMechanismRouting final {
public:
    TorchPreparedEntityMechanismRouting(
        const std::vector<std::vector<std::int64_t>> &entity_mechanisms,
        std::uint64_t entity_count, std::uint64_t mechanism_count,
        std::uint64_t owner_identity, std::uint64_t relation_id,
        std::uint64_t structure_id, std::uint64_t structure_epoch,
        std::uint64_t route_generation, std::uint64_t binding_generation,
        std::int64_t device)
        : device_(device) {
        TORCH_CHECK(device_ >= 0 && device_ < at::cuda::device_count(),
                    "prepared routing device index is invalid");
        identity_ = {owner_identity, relation_id, structure_id, structure_epoch,
                     route_generation, binding_generation};
        const auto status = ce_routing::prepare_entity_mechanism_index_v1(
            entity_mechanisms, entity_count, mechanism_count, identity_, &index_);
        switch (status) {
        case ce_routing::prepared_routing_status_v1::success:
            break;
        case ce_routing::prepared_routing_status_v1::mechanism_out_of_range:
            throw py::value_error("route table contains an out-of-range mechanism ID");
        case ce_routing::prepared_routing_status_v1::invalid_identity:
            throw py::value_error("prepared routing identity fields must be nonzero");
        case ce_routing::prepared_routing_status_v1::entity_count_mismatch:
            throw py::value_error("route row count does not match the entity domain");
        default:
            throw py::value_error("prepared route table is invalid or exceeds capacity");
        }

        c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_));
        auto stream = at::cuda::getCurrentCUDAStream(device_);
        device_terms_ = upload_vector(index_.device_terms,
            static_cast<std::int64_t>(index_.device_terms.size()), 2, device_);
        device_term_offsets_ = upload_offsets(index_.device_term_offsets, device_);
        device_items_ = upload_vector(index_.device_items,
            static_cast<std::int64_t>(index_.device_items.size()), 2, device_);
        const auto sync_status = cudaStreamSynchronize(stream.stream());
        TORCH_CHECK(sync_status == cudaSuccess,
                    "prepared routing projection upload failed: ",
                    cudaGetErrorString(sync_status));
    }

    py::tuple query(const TorchTensor &entity_ids,
                    std::uint64_t owner_identity, std::uint64_t relation_id,
                    std::uint64_t structure_id, std::uint64_t structure_epoch,
                    std::uint64_t route_generation,
                    std::uint64_t binding_generation) {
        TORCH_CHECK(owner_identity == identity_.owner_identity
                    && relation_id == identity_.relation_id
                    && structure_id == identity_.structure_id,
                    "prepared route owner or structure identity is stale");
        TORCH_CHECK(structure_epoch == identity_.structure_epoch,
                    "prepared route structure epoch is stale");
        TORCH_CHECK(route_generation == identity_.route_generation,
                    "prepared route value generation is stale");
        TORCH_CHECK(binding_generation == identity_.binding_generation,
                    "prepared route binding generation is stale");
        TORCH_CHECK(entity_ids.is_cuda() && entity_ids.get_device() == device_,
                    "entity IDs must be resident on the prepared CUDA device");
        TORCH_CHECK(entity_ids.dim() == 2
                    && (entity_ids.scalar_type() == at::kInt
                        || entity_ids.scalar_type() == at::kLong),
                    "entity IDs must be a rank-two int32 or int64 tensor");
        TORCH_CHECK(entity_ids.is_contiguous(),
                    "entity IDs must be contiguous; normalize explicitly in Python");
        TORCH_CHECK(!entity_ids.requires_grad(), "entity IDs cannot require gradients");

        const auto cells = static_cast<std::uint64_t>(entity_ids.size(0));
        const auto participants = static_cast<std::uint64_t>(entity_ids.size(1));
        const auto options = at::TensorOptions().device(at::Device(at::kCUDA,
            static_cast<c10::DeviceIndex>(device_)));
        if (cells == 0u) {
            auto empty_offsets = at::zeros({1}, options.dtype(at::kLong));
            auto empty_mechanisms = at::empty({0}, options.dtype(at::kInt));
            return py::make_tuple(empty_offsets, empty_mechanisms);
        }

        ce_routing::prepared_routing_query_requirements_v1 requirements{};
        const auto requirement_status = ce_routing::query_requirements_v1(
            index_, cells, participants, &requirements);
        TORCH_CHECK(requirement_status ==
                        ce_routing::prepared_routing_status_v1::success,
                    "prepared route query exceeds the supported shape/capacity");
        TORCH_CHECK(requirements.query_count <= static_cast<std::uint64_t>(
                        std::numeric_limits<std::int64_t>::max())
                    && requirements.output_item_capacity <= static_cast<std::uint64_t>(
                        std::numeric_limits<std::int64_t>::max()),
                    "prepared route query tensor dimensions exceed int64");

        c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_));
        const auto stream = at::cuda::getCurrentCUDAStream(device_);
        const auto workspace = workspace_for_stream(stream.stream());
        std::lock_guard<std::mutex> use_lock(workspace->use_mutex);
        reserve_workspace(*workspace, requirements, cells);

        auto output_offsets = allocate_cuda(
            {static_cast<std::int64_t>(cells + 1u)}, at::kLong, device_);
        auto output_mechanisms = TorchTensor{};
        const auto query_terms = requirements.query_count == 0u ? nullptr
            : reinterpret_cast<ce_indexing::posting_term_v1 *>(
                workspace->queries.data_ptr<std::int64_t>());
        const auto query_offsets = reinterpret_cast<std::uint64_t *>(
            workspace->query_offsets.data_ptr<std::int64_t>());
        auto *output_items = requirements.output_item_capacity == 0u ? nullptr
            : reinterpret_cast<ce_indexing::indexed_work_item_v1 *>(
                workspace->output_items.data_ptr<std::int64_t>());
        auto *cursors = requirements.query_count == 0u ? nullptr
            : reinterpret_cast<ce_indexing::cuda_posting_scratch_v1::cursor_v1 *>(
                workspace->cursors.data_ptr<std::int64_t>());
        auto *counts = reinterpret_cast<std::uint64_t *>(
            workspace->counts.data_ptr<std::int64_t>());
        auto *device_status = reinterpret_cast<ce_indexing::cuda_posting_status_v1 *>(
            workspace->device_status.data_ptr<std::int64_t>());
        auto device_index = index_.bind_device_view(
            reinterpret_cast<const ce_indexing::posting_term_v1 *>(
                device_terms_.data_ptr<std::int64_t>()),
            reinterpret_cast<const std::uint64_t *>(
                device_term_offsets_.data_ptr<std::int64_t>()),
            reinterpret_cast<const ce_indexing::indexed_work_item_v1 *>(
                device_items_.data_ptr<std::int64_t>()));
        const ce_indexing::cuda_posting_output_v1 output{
            reinterpret_cast<std::uint64_t *>(output_offsets.data_ptr<std::int64_t>()),
            cells + 1u, output_items, requirements.output_item_capacity};
        const ce_indexing::cuda_posting_scratch_v1 scratch{
            cursors, requirements.query_count, counts, device_status};

        const auto *input_i32 = entity_ids.scalar_type() == at::kInt
            ? entity_ids.data_ptr<std::int32_t>() : nullptr;
        const auto *input_i64 = entity_ids.scalar_type() == at::kLong
            ? entity_ids.data_ptr<std::int64_t>() : nullptr;
        cudaError_t launch_status = cudaErrorInvalidValue;
        if (input_i32 != nullptr) {
            launch_status = ce_routing::cuda_query_prepared_entity_mechanisms_i32_v1(
                device_index, identity_.structure_epoch, index_.entity_count,
                cells, participants, input_i32, output, scratch, query_terms,
                requirements.query_count, query_offsets, cells + 1u,
                stream.stream());
        } else {
            launch_status = ce_routing::cuda_query_prepared_entity_mechanisms_i64_v1(
                device_index, identity_.structure_epoch, index_.entity_count,
                cells, participants, input_i64, output, scratch, query_terms,
                requirements.query_count, query_offsets, cells + 1u,
                stream.stream());
        }
        TORCH_CHECK(launch_status == cudaSuccess,
                    "prepared route query launch failed: ",
                    cudaGetErrorString(launch_status));

        for (const auto &tensor : {entity_ids, device_terms_, device_term_offsets_,
                                   device_items_, workspace->queries,
                                   workspace->query_offsets, workspace->output_items,
                                   workspace->cursors, workspace->counts,
                                   workspace->device_status, output_offsets})
            record_stream(tensor, stream);

        const auto copy_status = cudaMemcpyAsync(workspace->host_status.data_ptr(),
            workspace->device_status.data_ptr(),
            sizeof(ce_indexing::cuda_posting_status_v1), cudaMemcpyDeviceToHost,
            stream.stream());
        TORCH_CHECK(copy_status == cudaSuccess,
                    "prepared route status copy failed: ",
                    cudaGetErrorString(copy_status));
        // The existing latent-transition op requires an exact-length mechanism
        // tensor. This one status read is the explicit dynamic-count boundary.
        const auto sync_status = cudaStreamSynchronize(stream.stream());
        TORCH_CHECK(sync_status == cudaSuccess,
                    "prepared route count synchronization failed: ",
                    cudaGetErrorString(sync_status));
        const auto status = *reinterpret_cast<const ce_indexing::cuda_posting_status_v1 *>(
            workspace->host_status.data_ptr<std::int64_t>());
        if (status.code == static_cast<std::uint32_t>(
                ce_indexing::cuda_posting_code_v1::output_overflow))
            throw std::overflow_error("prepared route workset exceeds its bounded capacity");
        if (status.code == static_cast<std::uint32_t>(
                ce_indexing::cuda_posting_code_v1::invalid_query))
            throw py::value_error("entity ID is outside the prepared routing domain");
        TORCH_CHECK(status.code == static_cast<std::uint32_t>(
                        ce_indexing::cuda_posting_code_v1::success),
                    "prepared route query failed with status code ", status.code);
        TORCH_CHECK(status.required <= requirements.output_item_capacity
                    && status.required <= static_cast<std::uint64_t>(
                        std::numeric_limits<std::int64_t>::max()),
                    "prepared route query returned an invalid output count");

        output_mechanisms = allocate_cuda(
            {static_cast<std::int64_t>(status.required)}, at::kInt, device_);
        const auto emit_status = ce_routing::cuda_emit_prepared_mechanism_indices_v1(
            output_items, status.required,
            status.required == 0u ? nullptr : output_mechanisms.data_ptr<std::int32_t>(),
            status.required, stream.stream());
        TORCH_CHECK(emit_status == cudaSuccess,
                    "prepared mechanism-ID emission failed: ",
                    cudaGetErrorString(emit_status));
        record_stream(workspace->output_items, stream);
        record_stream(output_mechanisms, stream);
        return py::make_tuple(output_offsets, output_mechanisms);
    }

private:
    std::shared_ptr<QueryWorkspace> workspace_for_stream(cudaStream_t stream) {
        const auto key = reinterpret_cast<std::uintptr_t>(stream);
        std::lock_guard<std::mutex> lock(workspaces_mutex_);
        auto &workspace = workspaces_[key];
        if (!workspace) workspace = std::make_shared<QueryWorkspace>();
        return workspace;
    }

    void reserve_workspace(QueryWorkspace &workspace,
                          const ce_routing::prepared_routing_query_requirements_v1 &req,
                          std::uint64_t cells) {
        const auto device_options = at::TensorOptions().device(at::Device(at::kCUDA,
            static_cast<c10::DeviceIndex>(device_)));
        if (workspace.query_capacity < req.query_count) {
            TORCH_CHECK(req.query_count <= static_cast<std::uint64_t>(
                            std::numeric_limits<std::int64_t>::max()),
                        "prepared query scratch exceeds int64");
            workspace.queries = at::empty(
                {static_cast<std::int64_t>(req.query_count), 2},
                device_options.dtype(at::kLong));
            workspace.cursors = at::empty(
                {static_cast<std::int64_t>(req.query_count), 4},
                device_options.dtype(at::kLong));
            workspace.query_capacity = req.query_count;
        }
        if (workspace.cell_capacity < cells) {
            TORCH_CHECK(cells <= static_cast<std::uint64_t>(
                            std::numeric_limits<std::int64_t>::max() - 1),
                        "prepared cell scratch exceeds int64");
            workspace.query_offsets = at::empty(
                {static_cast<std::int64_t>(cells + 1u)},
                device_options.dtype(at::kLong));
            workspace.counts = at::empty(
                {static_cast<std::int64_t>(cells)}, device_options.dtype(at::kLong));
            workspace.cell_capacity = cells;
        }
        if (workspace.output_capacity < req.output_item_capacity) {
            TORCH_CHECK(req.output_item_capacity <= static_cast<std::uint64_t>(
                            std::numeric_limits<std::int64_t>::max()),
                        "prepared output scratch exceeds int64");
            workspace.output_items = at::empty(
                {static_cast<std::int64_t>(req.output_item_capacity), 2},
                device_options.dtype(at::kLong));
            workspace.output_capacity = req.output_item_capacity;
        }
        if (!workspace.device_status.defined()) {
            workspace.device_status = at::empty({2}, device_options.dtype(at::kLong));
            workspace.host_status = at::empty({2}, at::TensorOptions()
                .dtype(at::kLong).device(at::kCPU).pinned_memory(true));
        }
    }

    ce_routing::prepared_routing_identity_v1 identity_{};
    ce_routing::prepared_entity_mechanism_index_v1 index_{};
    std::int64_t device_ = -1;
    TorchTensor device_terms_;
    TorchTensor device_term_offsets_;
    TorchTensor device_items_;
    std::mutex workspaces_mutex_;
    std::unordered_map<std::uintptr_t, std::shared_ptr<QueryWorkspace>> workspaces_;
};

} // namespace

void bind_sparse_support_prepared_routing(py::module_ &module) {
    py::class_<TorchPreparedEntityMechanismRouting,
               std::shared_ptr<TorchPreparedEntityMechanismRouting>>(
        module, "PreparedEntityMechanismRouting")
        .def(py::init<const std::vector<std::vector<std::int64_t>> &,
                      std::uint64_t, std::uint64_t, std::uint64_t,
                      std::uint64_t, std::uint64_t, std::uint64_t,
                      std::uint64_t, std::uint64_t, std::int64_t>(),
            py::arg("entity_mechanisms"), py::arg("entity_count"),
            py::arg("mechanism_count"), py::arg("owner_identity"),
            py::arg("relation_id"), py::arg("structure_id"),
            py::arg("structure_epoch"), py::arg("route_generation"),
            py::arg("binding_generation"), py::arg("device"))
        .def("query", &TorchPreparedEntityMechanismRouting::query,
            py::arg("entity_ids"), py::arg("owner_identity"),
            py::arg("relation_id"), py::arg("structure_id"),
            py::arg("structure_epoch"), py::arg("route_generation"),
            py::arg("binding_generation"));
}
