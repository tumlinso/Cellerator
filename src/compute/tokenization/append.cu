#include <Cellerator/compute/tokenization/sequence_buffer.hh>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>

namespace cellerator::tokenization {
namespace {

constexpr int kThreads = 256;

__global__ void validate_append_offsets(
    const std::int64_t* offsets, std::int64_t input_rows,
    std::int64_t source_size, unsigned long long* error) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= input_rows) return;
    if (offsets[i] < 0 || offsets[i + 1] < offsets[i]
        || (i == 0 && offsets[i] != 0)
        || (i + 1 == input_rows && offsets[i + 1] != source_size))
        atomicExch(error, 1ull);
}

__global__ void classify_append_rows(
    const std::int64_t* selected_rows,
    const std::int64_t* offsets,
    const std::int64_t* token_ids,
    const std::int64_t* row_owner,
    const unsigned long long* append_error,
    std::int64_t input_rows,
    std::int64_t source_size,
    const std::int64_t* lengths,
    const std::int64_t* starts,
    const std::int64_t* capacities,
    std::int64_t sequence_count,
    bool* succeeded) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= input_rows) return;
    const auto row = selected_rows[i];
    const auto begin = offsets[i];
    const auto end = offsets[i + 1];
    if (*append_error != 0 || row < 0 || row >= sequence_count
        || begin < 0 || end < begin || end > source_size) {
        succeeded[i] = false;
        return;
    }
    if (row_owner[row] != i) {
        succeeded[i] = false;
        return;
    }
    const auto count = end - begin;
    const auto current = lengths[row];
    const auto capacity = capacities[row];
    bool nonnegative = true;
    for (std::int64_t token = begin; token < end; ++token) {
        if (token_ids[token] < 0) {
            nonnegative = false;
            break;
        }
    }
    succeeded[i] = succeeded[i] && nonnegative
                && current >= 0 && current <= capacity
                && count <= capacity - (current >= 0 && current <= capacity ? current : capacity)
                && starts[row] >= 0 && starts[row] + capacities[row] >= starts[row];
}

__global__ void claim_append_rows(
    const std::int64_t* selected_rows,
    std::int64_t input_rows,
    std::int64_t sequence_count,
    std::int64_t* row_owner) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= input_rows) return;
    const auto row = selected_rows[i];
    if (row >= 0 && row < sequence_count)
        atomicMin(reinterpret_cast<unsigned long long*>(row_owner + row),
                  static_cast<unsigned long long>(i));
}

__global__ void append_copy_bytes(
    const std::int64_t* selected_rows,
    const std::int64_t* offsets,
    const bool* succeeded,
    const std::int64_t* lengths,
    const std::int64_t* starts,
    const std::uint8_t* source,
    std::uint8_t* destination,
    std::int64_t input_rows,
    std::int64_t source_tokens,
    std::int64_t elements_per_token,
    std::int64_t bytes_per_element) {
    const std::int64_t flat = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t total = source_tokens * elements_per_token;
    if (flat >= total) return;
    const auto token = flat / elements_per_token;
    const auto element = flat - token * elements_per_token;
    std::int64_t lo = 0;
    std::int64_t hi = input_rows;
    while (lo < hi) {
        const auto mid = lo + (hi - lo) / 2;
        if (offsets[mid + 1] <= token) lo = mid + 1;
        else hi = mid;
    }
    if (lo >= input_rows || !succeeded[lo]) return;
    const auto row = selected_rows[lo];
    const auto local = token - offsets[lo];
    const auto destination_token = starts[row] + lengths[row] + local;
    const auto src_byte = flat * bytes_per_element;
    const auto dst_byte = (destination_token * elements_per_token + element) * bytes_per_element;
    for (std::int64_t b = 0; b < bytes_per_element; ++b)
        destination[dst_byte + b] = source[src_byte + b];
}

__global__ void commit_append_lengths(
    const std::int64_t* selected_rows,
    const std::int64_t* offsets,
    const bool* succeeded,
    std::int64_t* lengths,
    std::int64_t input_rows) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= input_rows || !succeeded[i]) return;
    const auto row = selected_rows[i];
    lengths[row] += offsets[i + 1] - offsets[i];
}

void check_cuda_tensor(const at::Tensor& tensor, const at::Device& device, const char* name) {
    if (!tensor.is_cuda() || tensor.device() != device || !tensor.is_contiguous())
        throw std::invalid_argument(std::string(name) + " must be contiguous on the sequence-buffer device");
}

} // namespace

void launch_sequence_append_cuda(
    const at::Tensor& selected_rows, const at::Tensor& update_offsets,
    const at::Tensor& update_ids, at::Tensor& lengths,
    const at::Tensor& starts, const at::Tensor& capacities,
    const at::Tensor& row_owner_scratch, at::Tensor& append_error_scratch,
    at::Tensor& succeeded, at::Tensor& token_storage,
    const std::unordered_map<std::string, at::Tensor>& update_attributes,
    std::unordered_map<std::string, at::Tensor>& attribute_storage) {
    const auto device = lengths.device();
    const std::array<const at::Tensor*, 10> tensors_to_check{
        &selected_rows, &update_offsets, &update_ids, &lengths, &starts, &capacities,
        &row_owner_scratch, &append_error_scratch, &succeeded, &token_storage};
    for (const auto* tensor : tensors_to_check)
        check_cuda_tensor(*tensor, device, "append tensor");
    for (const auto& [name, tensor] : update_attributes) {
        (void)name;
        check_cuda_tensor(tensor, device, "update attribute");
    }
    c10::cuda::CUDAGuard guard(device);
    const auto stream = at::cuda::getCurrentCUDAStream(device.index());
    auto record_use = [&](const at::Tensor& tensor) {
        c10::cuda::CUDACachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
    };
    const std::array<const at::Tensor*, 10> tensors_to_record{
        &selected_rows, &update_offsets, &update_ids, &lengths, &starts, &capacities,
        &row_owner_scratch, &append_error_scratch, &succeeded, &token_storage};
    for (const auto* tensor : tensors_to_record)
        record_use(*tensor);
    for (const auto& [name, tensor] : update_attributes) {
        (void)name;
        record_use(tensor);
        record_use(attribute_storage.at(name));
    }
    const auto rows = selected_rows.numel();
    if (rows == 0) return;
    const auto blocks = static_cast<unsigned int>((rows + kThreads - 1) / kThreads);
    validate_append_offsets<<<blocks, kThreads, 0, stream>>>(
        update_offsets.data_ptr<std::int64_t>(), rows, update_ids.numel(),
        reinterpret_cast<unsigned long long*>(append_error_scratch.data_ptr<std::int64_t>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    claim_append_rows<<<blocks, kThreads, 0, stream>>>(
        selected_rows.data_ptr<std::int64_t>(), rows, lengths.numel(),
        row_owner_scratch.data_ptr<std::int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    classify_append_rows<<<blocks, kThreads, 0, stream>>>(
        selected_rows.data_ptr<std::int64_t>(), update_offsets.data_ptr<std::int64_t>(),
        update_ids.data_ptr<std::int64_t>(),
        row_owner_scratch.data_ptr<std::int64_t>(),
        reinterpret_cast<const unsigned long long*>(append_error_scratch.data_ptr<std::int64_t>()),
        rows, update_ids.numel(), lengths.data_ptr<std::int64_t>(),
        starts.data_ptr<std::int64_t>(), capacities.data_ptr<std::int64_t>(),
        lengths.numel(), succeeded.data_ptr<bool>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto launch_copy = [&](const at::Tensor& source, at::Tensor& destination,
                           std::int64_t width) {
        if (source.numel() == 0 || width == 0) return;
        const auto elements = source.numel();
        const auto copy_blocks = static_cast<unsigned int>((elements + kThreads - 1) / kThreads);
        append_copy_bytes<<<copy_blocks, kThreads, 0, stream>>>(
            selected_rows.data_ptr<std::int64_t>(), update_offsets.data_ptr<std::int64_t>(),
            succeeded.data_ptr<bool>(), lengths.data_ptr<std::int64_t>(),
            starts.data_ptr<std::int64_t>(),
            reinterpret_cast<const std::uint8_t*>(source.data_ptr()),
            reinterpret_cast<std::uint8_t*>(destination.data_ptr()), rows,
            update_ids.numel(), width, source.element_size());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    launch_copy(update_ids, token_storage, 1);
    for (const auto& [name, source] : update_attributes) {
        auto& destination = attribute_storage.at(name);
        const auto width = source.numel() / (update_ids.numel() ? update_ids.numel() : 1);
        if (source.numel() != 0 && width == 0)
            throw std::invalid_argument("attribute storage width is invalid");
        launch_copy(source, destination, width);
    }
    commit_append_lengths<<<blocks, kThreads, 0, stream>>>(
        selected_rows.data_ptr<std::int64_t>(), update_offsets.data_ptr<std::int64_t>(),
        succeeded.data_ptr<bool>(), lengths.data_ptr<std::int64_t>(), rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace cellerator::tokenization
