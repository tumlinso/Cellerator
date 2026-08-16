/*
 * Initial correctness kernel validation (2026-08-14, Cellerator 1ebb734):
 * reference is build_gene_support_bitsets_cpu; target is Tesla V100 (sm_70);
 * shapes cover 0, 31, 32, and 33 sampled cells plus duplicate entries and
 * empty rows/genes. Command: ./build-sampling/geneSupportBitsetRuntimeTest.
 * Acceptance is bit-exact support words and detected-cell counts. This is a
 * correctness-first thread-per-cell atomicOr kernel; no performance advantage
 * over a maintained library path is claimed because no library primitive has
 * this CSR-to-gene-major-bitset operation. The opt-in 65,536 x 30,000 device
 * allocation smoke command is the same executable with --full-size-gpu.
 */

#include <Cellerator/compute/gene_support_bitset.hh>

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <new>
#include <utility>

namespace cellerator::compute::gene_support {

namespace {

namespace cs = ::cellerator::compute::sampling;
namespace ct = ::cellerator::types;

void set_error(std::string *error, const std::string &message) {
    if (error != nullptr) *error = message;
}

bool checked_multiply(std::size_t lhs,
                      std::size_t rhs,
                      std::size_t *out) {
    if (out == nullptr) return false;
    if (lhs != 0u && rhs > std::numeric_limits<std::size_t>::max() / lhs) return false;
    *out = lhs * rhs;
    return true;
}

bool validate_sampled_csr(const cs::sampled_csr_structure_view &sampled,
                          std::string *error) {
    if (sampled.provenance == nullptr) {
        set_error(error, "sampled CSR provenance is null");
        return false;
    }
    if (sampled.provenance->hash_algorithm != cs::splitmix64_algorithm_name
        || sampled.provenance->hash_version != cs::splitmix64_algorithm_version) {
        set_error(error, "sampled CSR uses an unsupported hash contract");
        return false;
    }
    if (sampled.provenance->selected_rows != sampled.sampled_row_count) {
        set_error(error, "sampled CSR row count does not match provenance");
        return false;
    }
    if (sampled.sampled_row_count > sampled.provenance->total_rows) {
        set_error(error, "sampled CSR row count exceeds its population");
        return false;
    }
    if (sampled.sampled_row_count > (std::uint64_t) std::numeric_limits<ct::dim_t>::max()
        || sampled.gene_count > (std::uint64_t) std::numeric_limits<ct::idx_t>::max()
        || sampled.nnz > (std::uint64_t) std::numeric_limits<ct::ptr_t>::max()) {
        set_error(error, "sampled CSR dimensions exceed Cellerator canonical index limits");
        return false;
    }
    if (sampled.row_ptr == nullptr) {
        set_error(error, "sampled CSR row pointers are null");
        return false;
    }
    if (sampled.nnz != 0u && sampled.column_indices == nullptr) {
        set_error(error, "sampled CSR column indices are null");
        return false;
    }
    if (sampled.sampled_row_count != 0u && sampled.sampled_position_to_global_row == nullptr) {
        set_error(error, "sampled-position/global-row mapping is null");
        return false;
    }
    if (sampled.row_ptr[0] != 0u) {
        set_error(error, "sampled CSR row pointers must begin at zero");
        return false;
    }

    ct::ptr_t previous = 0u;
    for (std::uint64_t row = 0u; row < sampled.sampled_row_count; ++row) {
        const ct::ptr_t next = sampled.row_ptr[(std::size_t) row + 1u];
        if (next < previous || (std::uint64_t) next > sampled.nnz) {
            set_error(error, "sampled CSR row pointers are non-monotonic or exceed nnz");
            return false;
        }
        const std::uint64_t global_row = sampled.sampled_position_to_global_row[row];
        if (global_row >= sampled.provenance->total_rows
            || (row != 0u && global_row <= sampled.sampled_position_to_global_row[row - 1u])) {
            set_error(error, "sampled-position/global-row mapping is not strictly ascending and in range");
            return false;
        }
        previous = next;
    }
    if ((std::uint64_t) previous != sampled.nnz) {
        set_error(error, "sampled CSR terminal row pointer does not equal nnz");
        return false;
    }
    for (std::uint64_t slot = 0u; slot < sampled.nnz; ++slot) {
        if ((std::uint64_t) sampled.column_indices[slot] >= sampled.gene_count) {
            set_error(error, "sampled CSR contains an out-of-range gene index");
            return false;
        }
    }
    return true;
}

bool allocate_host_result(const cs::sampled_csr_structure_view &sampled,
                          const gene_support_layout &layout,
                          owned_gene_support_bitsets *result,
                          std::string *error) {
    std::unique_ptr<support_word_t[]> support;
    std::unique_ptr<ct::count_value_t[]> counts;
    std::unique_ptr<std::uint64_t[]> global_rows;
    if (layout.support_word_count != 0u) {
        support.reset(new (std::nothrow) support_word_t[layout.support_word_count]());
    }
    if (layout.gene_count != 0u) {
        counts.reset(new (std::nothrow) ct::count_value_t[(std::size_t) layout.gene_count]());
    }
    if (layout.sampled_cell_count != 0u) {
        global_rows.reset(new (std::nothrow) std::uint64_t[(std::size_t) layout.sampled_cell_count]);
    }
    if ((layout.support_word_count != 0u && support == nullptr)
        || (layout.gene_count != 0u && counts == nullptr)
        || (layout.sampled_cell_count != 0u && global_rows == nullptr)) {
        set_error(error, "failed to allocate owned gene-support output");
        return false;
    }
    if (layout.sampled_cell_count != 0u) {
        std::copy_n(sampled.sampled_position_to_global_row,
                    (std::size_t) layout.sampled_cell_count,
                    global_rows.get());
    }
    *result = owned_gene_support_bitsets(
        layout, std::move(support), std::move(counts), std::move(global_rows), *sampled.provenance);
    return true;
}

bool cuda_status(cudaError_t status, const char *operation, std::string *error) {
    if (status == cudaSuccess) return true;
    set_error(error, std::string(operation) + ": " + cudaGetErrorString(status));
    return false;
}

struct device_buffers {
    ct::ptr_t *row_ptr = nullptr;
    ct::idx_t *column_indices = nullptr;
    support_word_t *support = nullptr;
    ct::count_value_t *counts = nullptr;
    ct::u32 *invalid_input = nullptr;
};

cudaError_t free_device_buffers(device_buffers *buffers) {
    if (buffers == nullptr) return cudaSuccess;
    cudaError_t first_error = cudaSuccess;
    auto release = [&](void *pointer) {
        if (pointer == nullptr) return;
        const cudaError_t status = cudaFree(pointer);
        if (first_error == cudaSuccess && status != cudaSuccess) first_error = status;
    };
    release(buffers->invalid_input);
    release(buffers->counts);
    release(buffers->support);
    release(buffers->column_indices);
    release(buffers->row_ptr);
    *buffers = {};
    return first_error;
}

__global__ void build_gene_support_kernel(const ct::ptr_t *row_ptr,
                                          const ct::idx_t *column_indices,
                                          ct::dim_t sampled_cell_count,
                                          ct::idx_t gene_count,
                                          ct::ptr_t nnz,
                                          std::size_t words_per_gene,
                                          support_word_t *gene_support,
                                          ct::count_value_t *detected_cell_counts,
                                          ct::u32 *invalid_input) {
    const ct::dim_t cell = (ct::dim_t) (blockIdx.x * blockDim.x + threadIdx.x);
    if (cell >= sampled_cell_count) return;
    const ct::ptr_t begin = row_ptr[cell];
    const ct::ptr_t end = row_ptr[cell + 1u];
    if (end < begin || end > nnz) {
        atomicOr(invalid_input, 1u);
        return;
    }
    const std::size_t word = (std::size_t) cell / cells_per_support_word;
    const support_word_t bit = (support_word_t) 1u << (cell % cells_per_support_word);
    for (ct::ptr_t slot = begin; slot < end; ++slot) {
        const ct::idx_t gene = column_indices[slot];
        if (gene >= gene_count) {
            atomicOr(invalid_input, 2u);
            continue;
        }
        support_word_t *destination = gene_support + (std::size_t) gene * words_per_gene + word;
        const support_word_t previous = atomicOr(destination, bit);
        if ((previous & bit) == 0u) atomicAdd(detected_cell_counts + gene, 1u);
    }
}

} // namespace

bool calculate_gene_support_layout(std::uint64_t sampled_cell_count,
                                   std::uint64_t gene_count,
                                   gene_support_layout *out,
                                   std::string *error) {
    if (out == nullptr) {
        set_error(error, "gene-support layout output is null");
        return false;
    }
    const std::uint64_t words64 = sampled_cell_count / cells_per_support_word
        + (sampled_cell_count % cells_per_support_word != 0u ? 1u : 0u);
    if (words64 > (std::uint64_t) std::numeric_limits<std::size_t>::max()
        || gene_count > (std::uint64_t) std::numeric_limits<std::size_t>::max()) {
        set_error(error, "gene-support dimensions exceed size_t");
        return false;
    }
    gene_support_layout layout;
    layout.sampled_cell_count = sampled_cell_count;
    layout.gene_count = gene_count;
    layout.words_per_gene = (std::size_t) words64;
    if (!checked_multiply((std::size_t) gene_count,
                          layout.words_per_gene,
                          &layout.support_word_count)
        || !checked_multiply(layout.support_word_count,
                             sizeof(support_word_t),
                             &layout.support_bytes)
        || !checked_multiply((std::size_t) gene_count,
                             sizeof(ct::count_value_t),
                             &layout.detection_count_bytes)) {
        set_error(error, "gene-support allocation size overflows size_t");
        return false;
    }
    *out = layout;
    return true;
}

owned_gene_support_bitsets::owned_gene_support_bitsets(
    gene_support_layout layout,
    std::unique_ptr<support_word_t[]> gene_support,
    std::unique_ptr<ct::count_value_t[]> detected_cell_counts,
    std::unique_ptr<std::uint64_t[]> sampled_position_to_global_row,
    cs::sample_provenance provenance) noexcept
    : layout_(layout),
      gene_support_(std::move(gene_support)),
      detected_cell_counts_(std::move(detected_cell_counts)),
      sampled_position_to_global_row_(std::move(sampled_position_to_global_row)),
      provenance_(std::move(provenance)) {}

gene_support_bitset_view owned_gene_support_bitsets::view() const noexcept {
    return {layout_, gene_support_.get(), detected_cell_counts_.get(),
            sampled_position_to_global_row_.get(), &provenance_};
}

const cs::sample_provenance &owned_gene_support_bitsets::sampling_provenance() const noexcept {
    return provenance_;
}

bool build_gene_support_bitsets_cpu(const cs::sampled_csr_structure_view &sampled,
                                    owned_gene_support_bitsets *out,
                                    std::string *error) {
    if (out == nullptr) {
        set_error(error, "owned gene-support output is null");
        return false;
    }
    if (!validate_sampled_csr(sampled, error)) return false;
    gene_support_layout layout;
    if (!calculate_gene_support_layout(sampled.sampled_row_count, sampled.gene_count, &layout, error)) {
        return false;
    }
    owned_gene_support_bitsets staged;
    if (!allocate_host_result(sampled, layout, &staged, error)) return false;
    for (std::uint64_t cell = 0u; cell < sampled.sampled_row_count; ++cell) {
        const std::size_t word = (std::size_t) cell / cells_per_support_word;
        const support_word_t bit = (support_word_t) 1u << (cell % cells_per_support_word);
        const ct::ptr_t begin = sampled.row_ptr[cell];
        const ct::ptr_t end = sampled.row_ptr[cell + 1u];
        for (ct::ptr_t slot = begin; slot < end; ++slot) {
            const ct::idx_t gene = sampled.column_indices[slot];
            support_word_t &destination = staged.gene_support_[(std::size_t) gene * layout.words_per_gene + word];
            if ((destination & bit) == 0u) {
                destination |= bit;
                ++staged.detected_cell_counts_[gene];
            }
        }
    }
    *out = std::move(staged);
    return true;
}

bool build_gene_support_bitsets_cuda(const cs::sampled_csr_structure_view &sampled,
                                     int device,
                                     owned_gene_support_bitsets *out,
                                     std::string *error) {
    if (out == nullptr) {
        set_error(error, "owned gene-support output is null");
        return false;
    }
    if (!validate_sampled_csr(sampled, error)) return false;
    gene_support_layout layout;
    if (!calculate_gene_support_layout(sampled.sampled_row_count, sampled.gene_count, &layout, error)) {
        return false;
    }
    owned_gene_support_bitsets staged;
    if (!allocate_host_result(sampled, layout, &staged, error)) return false;
    if (sampled.sampled_row_count == 0u || sampled.gene_count == 0u) {
        *out = std::move(staged);
        return true;
    }

    std::size_t row_ptr_bytes = 0u, column_index_bytes = 0u;
    if (sampled.sampled_row_count == std::numeric_limits<std::uint64_t>::max()
        || !checked_multiply((std::size_t) sampled.sampled_row_count + 1u,
                             sizeof(ct::ptr_t), &row_ptr_bytes)
        || !checked_multiply((std::size_t) sampled.nnz,
                             sizeof(ct::idx_t), &column_index_bytes)) {
        set_error(error, "sampled CSR device upload size overflows size_t");
        return false;
    }

    int device_count = 0, previous_device = 0;
    if (!cuda_status(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", error)) return false;
    if (device < 0 || device >= device_count) {
        set_error(error, "requested CUDA device is unavailable");
        return false;
    }
    if (!cuda_status(cudaGetDevice(&previous_device), "cudaGetDevice", error)) return false;
    if (!cuda_status(cudaSetDevice(device), "cudaSetDevice", error)) return false;

    device_buffers buffers;
    bool ok = true;
    auto fail_cuda = [&](cudaError_t status, const char *operation) {
        if (ok && status != cudaSuccess) ok = cuda_status(status, operation, error);
    };
    fail_cuda(cudaMalloc((void **) &buffers.row_ptr, row_ptr_bytes), "cudaMalloc(row_ptr)");
    if (ok && column_index_bytes != 0u) {
        fail_cuda(cudaMalloc((void **) &buffers.column_indices, column_index_bytes),
                  "cudaMalloc(column_indices)");
    }
    if (ok && layout.support_bytes != 0u) {
        fail_cuda(cudaMalloc((void **) &buffers.support, layout.support_bytes),
                  "cudaMalloc(gene_support)");
    }
    if (ok && layout.detection_count_bytes != 0u) {
        fail_cuda(cudaMalloc((void **) &buffers.counts, layout.detection_count_bytes),
                  "cudaMalloc(detected_cell_counts)");
    }
    if (ok) fail_cuda(cudaMalloc((void **) &buffers.invalid_input, sizeof(ct::u32)),
                              "cudaMalloc(invalid_input)");
    if (ok) fail_cuda(cudaMemcpy(buffers.row_ptr, sampled.row_ptr, row_ptr_bytes,
                                 cudaMemcpyHostToDevice), "cudaMemcpy(row_ptr H2D)");
    if (ok && column_index_bytes != 0u) {
        fail_cuda(cudaMemcpy(buffers.column_indices, sampled.column_indices, column_index_bytes,
                             cudaMemcpyHostToDevice), "cudaMemcpy(column_indices H2D)");
    }
    if (ok && layout.support_bytes != 0u) {
        fail_cuda(cudaMemset(buffers.support, 0, layout.support_bytes), "cudaMemset(gene_support)");
    }
    if (ok && layout.detection_count_bytes != 0u) {
        fail_cuda(cudaMemset(buffers.counts, 0, layout.detection_count_bytes),
                  "cudaMemset(detected_cell_counts)");
    }
    if (ok) fail_cuda(cudaMemset(buffers.invalid_input, 0, sizeof(ct::u32)),
                              "cudaMemset(invalid_input)");

    if (ok) {
        constexpr unsigned int block_size = 256u;
        const unsigned int grid_size =
            ((unsigned int) sampled.sampled_row_count + block_size - 1u) / block_size;
        build_gene_support_kernel<<<grid_size, block_size>>>(
            buffers.row_ptr,
            buffers.column_indices,
            (ct::dim_t) sampled.sampled_row_count,
            (ct::idx_t) sampled.gene_count,
            (ct::ptr_t) sampled.nnz,
            layout.words_per_gene,
            buffers.support,
            buffers.counts,
            buffers.invalid_input);
        fail_cuda(cudaGetLastError(), "build_gene_support_kernel launch");
    }
    if (ok) fail_cuda(cudaDeviceSynchronize(), "build_gene_support_kernel execution");

    ct::u32 invalid_input = 0u;
    if (ok) fail_cuda(cudaMemcpy(&invalid_input, buffers.invalid_input, sizeof(invalid_input),
                                 cudaMemcpyDeviceToHost), "cudaMemcpy(invalid_input D2H)");
    if (ok && invalid_input != 0u) {
        set_error(error, "CUDA gene-support kernel rejected invalid CSR input");
        ok = false;
    }
    if (ok && layout.support_bytes != 0u) {
        fail_cuda(cudaMemcpy(staged.gene_support_.get(), buffers.support, layout.support_bytes,
                             cudaMemcpyDeviceToHost), "cudaMemcpy(gene_support D2H)");
    }
    if (ok && layout.detection_count_bytes != 0u) {
        fail_cuda(cudaMemcpy(staged.detected_cell_counts_.get(), buffers.counts,
                             layout.detection_count_bytes, cudaMemcpyDeviceToHost),
                  "cudaMemcpy(detected_cell_counts D2H)");
    }

    const cudaError_t free_status = free_device_buffers(&buffers);
    if (ok && free_status != cudaSuccess) {
        ok = cuda_status(free_status, "cudaFree(gene-support buffers)", error);
    }
    if (previous_device != device) {
        const cudaError_t restore_status = cudaSetDevice(previous_device);
        if (ok && restore_status != cudaSuccess) {
            ok = cuda_status(restore_status, "cudaSetDevice(restore)", error);
        }
    }
    if (!ok) return false;
    *out = std::move(staged);
    return true;
}

} // namespace cellerator::compute::gene_support
