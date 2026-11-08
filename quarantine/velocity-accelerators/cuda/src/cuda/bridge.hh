#pragma once

#include <anndata/device_abi.h>
#include <cstddef>
#include <cstdint>

namespace velocity_cuda_bridge {
using anndata::device::DenseView;
using anndata::device::SparseView;
using anndata::device::ScalarType;

struct Matrix {
    std::uint32_t sparse = 0;
    DenseView dense{};
    SparseView compressed{};
};
struct Edge { std::uint64_t index, position; };
struct PreparedMatrix {
    Matrix matrix;
    const std::uint64_t* offsets = nullptr;
    const Edge* edges = nullptr;
};
struct Options {
    double percentile_low = 5, percentile_high = 95;
    double ratio_min, ratio_max, min_r2 = .01, min_ratio = .01;
    std::uint32_t trimming = 2, fit_intercept = 0, positive_intercept = 1;
    std::uint32_t r2_adjusted = 0, low_signal = 0, second_intercept = 0, refit_all_if_none = 0;
};
struct Result {
    void* initial_gamma = nullptr;
    void* initial_offset = nullptr;
    void* r2 = nullptr;
    std::uint8_t* selected = nullptr;
    std::uint8_t* initial_status = nullptr;
    void* gamma = nullptr;
    void* offset = nullptr;
    void* offset2 = nullptr;
    void* scale1 = nullptr;
    void* scale2 = nullptr;
    void* objective = nullptr;
    std::uint8_t* status = nullptr;
    std::uint8_t* low_signal = nullptr;
    std::uint8_t* refit_all = nullptr;
};
struct Fields {
    void* source_s = nullptr;
    void* source_u = nullptr;
    void* ms = nullptr;
    void* mu = nullptr;
    void* mss = nullptr;
    void* mus = nullptr;
    void* muu = nullptr;
    void* scores = nullptr;
    std::uint8_t* mask = nullptr;
};
struct Outputs {
    DenseView velocity{}, second_residual{}, spliced{}, unspliced{};
    DenseView spliced_second{}, cross{}, unspliced_second{};
    std::uint32_t block_local = 0;
};

// These calls enqueue only. Device allocation and host-visible completion are
// acquisition/caller boundaries in the module wrapper.
int prepare_scratch(std::uint64_t major, std::size_t* bytes);
int validate(Matrix, bool positive_graph, int* device_error, void* stream);
int prepare(Matrix, bool by_rows, std::uint64_t* offsets, Edge*, std::uint64_t* counts,
            void* scratch, std::size_t scratch_bytes, void* stream);
int pack(PreparedMatrix, PreparedMatrix, const std::uint64_t* genes,
         std::uint64_t first, std::uint64_t count, std::uint64_t rows,
         Fields, bool fp32, void* stream);
int pack_override(PreparedMatrix, PreparedMatrix, std::uint64_t source_gene,
                  std::uint64_t destination_column, std::uint64_t rows,
                  Fields, bool fp32, void* stream);
int moments(PreparedMatrix graph, std::uint64_t rows, std::uint64_t count,
            Fields, bool fp32, bool second, void* stream);
int initial_fit(std::uint64_t rows, std::uint64_t first, std::uint64_t count,
                const std::uint64_t* genes, DenseView fit_mask, DenseView gene_mask,
                Fields, Result, Options, bool fp32, void* stream);
int resolve(std::uint64_t genes, const std::uint64_t* source_genes, DenseView gene_mask,
            Result, Options, void* scores, bool fp32, bool stochastic, void* stream);
int scales(std::uint64_t rows, std::uint64_t first, std::uint64_t count,
           const std::uint64_t* genes, DenseView fit_mask, DenseView gene_mask,
           Fields, Result, Options, bool fp32, void* stream);
int fit(std::uint64_t rows, std::uint64_t first, std::uint64_t count,
        const std::uint64_t* genes, DenseView fit_mask, DenseView gene_mask,
        Fields, Result, Options, bool fp32, void* stream);
int emit(std::uint64_t rows, std::uint64_t first, std::uint64_t count,
         Fields, Result, Outputs, bool fp32, bool stochastic, void* stream);
int neighborhood(PreparedMatrix, DenseView output, void* stream);
}
