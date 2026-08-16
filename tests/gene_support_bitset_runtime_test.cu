#include <Cellerator/compute/gene_support_bitset.hh>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace cg = ::cellerator::compute::gene_support;
namespace cm = ::cellerator::matrix;
namespace cs = ::cellerator::compute::sampling;
namespace ct = ::cellerator::types;

int require(bool ok, const char *label) {
    if (!ok) std::fprintf(stderr, "%s\n", label);
    return ok ? 1 : 0;
}

bool fill_patterned_source(ct::dim_t rows, ct::dim_t genes, cm::compressed *source) {
    std::vector<ct::ptr_t> row_ptr((std::size_t) rows + 1u, 0u);
    std::vector<ct::idx_t> columns;
    for (ct::dim_t row = 0u; row < rows; ++row) {
        if (row % 7u != 1u) {
            const ct::idx_t primary = row % (genes - 1u);
            columns.push_back(primary);
            if (row % 5u == 0u) columns.push_back(primary); // duplicate is intentional
            if (row % 3u == 0u) columns.push_back((primary + 2u) % (genes - 1u));
        }
        row_ptr[(std::size_t) row + 1u] = (ct::ptr_t) columns.size();
    }
    cm::init(source, rows, genes, (ct::nnz_t) columns.size(), cm::compressed_by_row);
    if (!cm::allocate(source)) return false;
    std::copy(row_ptr.begin(), row_ptr.end(), source->majorPtr);
    for (std::size_t slot = 0u; slot < columns.size(); ++slot) {
        source->minorIdx[slot] = columns[slot];
        source->val[slot] = __float2half((float) slot + 1.0f);
    }
    return true;
}

bool build_exact_sample(const cm::compressed &source,
                        std::uint64_t requested,
                        const char *split,
                        cs::owned_sampled_csr_structure *sampled,
                        std::string *error) {
    cs::sample_spec spec;
    cs::cell_identity_view identities;
    cs::sample_plan plan;
    spec.mode = cs::selection_mode::exact_lowest_hash;
    spec.seed = 0x123456789abcdef0ull;
    spec.split_name = split;
    spec.requested_row_count = requested;
    return cs::build_sample_plan(source.rows, spec, identities, &plan, error)
        && cs::materialize_sampled_csr_structure(&source, plan, sampled, error);
}

void reconstruct_expected(const cs::sampled_csr_structure_view &sampled,
                          const cg::gene_support_layout &layout,
                          std::vector<cg::support_word_t> *support,
                          std::vector<ct::count_value_t> *counts) {
    support->assign(layout.support_word_count, 0u);
    counts->assign((std::size_t) layout.gene_count, 0u);
    for (std::uint64_t cell = 0u; cell < sampled.sampled_row_count; ++cell) {
        const std::size_t word = (std::size_t) cell / cg::cells_per_support_word;
        const cg::support_word_t bit = 1u << (cell % cg::cells_per_support_word);
        for (ct::ptr_t slot = sampled.row_ptr[cell]; slot < sampled.row_ptr[cell + 1u]; ++slot) {
            const ct::idx_t gene = sampled.column_indices[slot];
            cg::support_word_t &destination =
                (*support)[(std::size_t) gene * layout.words_per_gene + word];
            if ((destination & bit) == 0u) {
                destination |= bit;
                ++(*counts)[gene];
            }
        }
    }
}

bool matches_expected(const cg::gene_support_bitset_view &actual,
                      const std::vector<cg::support_word_t> &support,
                      const std::vector<ct::count_value_t> &counts) {
    if (actual.layout.support_word_count != support.size()
        || actual.layout.gene_count != counts.size()) return false;
    for (std::size_t i = 0u; i < support.size(); ++i) {
        if (actual.gene_support[i] != support[i]) return false;
    }
    for (std::size_t i = 0u; i < counts.size(); ++i) {
        if (actual.detected_cell_counts[i] != counts[i]) return false;
    }
    return true;
}

bool results_equal(const cg::gene_support_bitset_view &lhs,
                   const cg::gene_support_bitset_view &rhs) {
    if (lhs.layout.sampled_cell_count != rhs.layout.sampled_cell_count
        || lhs.layout.gene_count != rhs.layout.gene_count
        || lhs.layout.words_per_gene != rhs.layout.words_per_gene
        || lhs.layout.support_word_count != rhs.layout.support_word_count) return false;
    for (std::size_t i = 0u; i < lhs.layout.support_word_count; ++i) {
        if (lhs.gene_support[i] != rhs.gene_support[i]) return false;
    }
    for (std::size_t i = 0u; i < lhs.layout.gene_count; ++i) {
        if (lhs.detected_cell_counts[i] != rhs.detected_cell_counts[i]) return false;
    }
    for (std::size_t i = 0u; i < lhs.layout.sampled_cell_count; ++i) {
        if (lhs.sampled_position_to_global_row[i] != rhs.sampled_position_to_global_row[i]) return false;
    }
    return lhs.provenance != nullptr && rhs.provenance != nullptr
        && lhs.provenance->seed == rhs.provenance->seed
        && lhs.provenance->hash_version == rhs.provenance->hash_version
        && lhs.provenance->split_name == rhs.provenance->split_name;
}

int test_sample_count_boundary(std::uint64_t cell_count, bool gpu_available) {
    cm::compressed source;
    cs::owned_sampled_csr_structure sampled;
    cg::owned_gene_support_bitsets cpu, gpu;
    std::vector<cg::support_word_t> expected_support;
    std::vector<ct::count_value_t> expected_counts;
    std::string error;
    cm::init(&source);
    if (!require(fill_patterned_source((ct::dim_t) cell_count, 7u, &source),
                 "failed to allocate patterned CSR fixture")) return 10;
    if (!require(build_exact_sample(source, cell_count, "support-boundary", &sampled, &error),
                 error.c_str())) return 11;
    const cs::sampled_csr_structure_view sampled_view = sampled.view();
    if (!require(cg::build_gene_support_bitsets_cpu(sampled_view, &cpu, &error), error.c_str())) return 12;
    const cg::gene_support_bitset_view cpu_view = cpu.view();
    const std::size_t expected_words = (std::size_t) ((cell_count + 31u) / 32u);
    if (!require(cpu_view.layout.words_per_gene == expected_words,
                 "words_per_gene is incorrect at a 32-cell boundary")) return 13;
    reconstruct_expected(sampled_view, cpu_view.layout, &expected_support, &expected_counts);
    if (!require(matches_expected(cpu_view, expected_support, expected_counts),
                 "CPU support bitsets do not exactly reconstruct sampled CSR")) return 14;
    if (!require(cpu_view.detected_cell_counts[6u] == 0u,
                 "structurally empty gene has a nonzero detection count")) return 15;
    for (std::size_t word = 0u; word < cpu_view.layout.words_per_gene; ++word) {
        if (!require(cpu_view.gene_support[6u * cpu_view.layout.words_per_gene + word] == 0u,
                     "structurally empty gene has support bits")) return 16;
    }
    if (!require(cpu_view.sampled_position_to_global_row[0u] == 0u
                 && cpu_view.sampled_position_to_global_row[cell_count - 1u] == cell_count - 1u,
                 "sampled cell positions do not preserve global-row identity")) return 17;
    if (!require(cpu_view.provenance == &cpu.sampling_provenance()
                 && cpu_view.provenance->selected_rows == cell_count,
                 "gene-support result did not retain immutable sampling provenance")) return 18;
    if (gpu_available) {
        if (!require(cg::build_gene_support_bitsets_cuda(sampled_view, 0, &gpu, &error), error.c_str())) return 19;
        if (!require(results_equal(cpu_view, gpu.view()), "CPU/GPU gene-support results differ")) return 20;
    }
    cm::clear(&source);
    return 0;
}

int test_deterministic_subset_alignment(bool gpu_available) {
    cm::compressed source;
    cs::owned_sampled_csr_structure first_sample, second_sample;
    cg::owned_gene_support_bitsets first, second, gpu;
    std::string error;
    cm::init(&source);
    if (!require(fill_patterned_source(48u, 7u, &source), "failed to allocate subset CSR fixture")) return 30;
    if (!require(build_exact_sample(source, 33u, "support-subset", &first_sample, &error), error.c_str())) return 31;
    if (!require(build_exact_sample(source, 33u, "support-subset", &second_sample, &error), error.c_str())) return 32;
    if (!require(cg::build_gene_support_bitsets_cpu(first_sample.view(), &first, &error), error.c_str())) return 33;
    if (!require(cg::build_gene_support_bitsets_cpu(second_sample.view(), &second, &error), error.c_str())) return 34;
    const cg::gene_support_bitset_view view = first.view();
    if (!require(results_equal(view, second.view()), "repeated sample-to-bitset flow is not deterministic")) return 35;
    bool non_identity_position = false;
    for (std::uint64_t cell = 0u; cell < view.layout.sampled_cell_count; ++cell) {
        const std::uint64_t global_row = view.sampled_position_to_global_row[cell];
        if (global_row != cell) non_identity_position = true;
        const cg::support_word_t bit = 1u << (cell % 32u);
        const std::size_t word = (std::size_t) cell / 32u;
        for (ct::ptr_t slot = source.majorPtr[global_row]; slot < source.majorPtr[global_row + 1u]; ++slot) {
            const ct::idx_t gene = source.minorIdx[slot];
            if (!require((view.gene_support[(std::size_t) gene * view.layout.words_per_gene + word] & bit) != 0u,
                         "sampled position bit is not aligned to its global source row")) return 36;
        }
    }
    if (!require(non_identity_position, "subset fixture did not exercise non-identity global-row mapping")) return 37;
    if (gpu_available) {
        if (!require(cg::build_gene_support_bitsets_cuda(first_sample.view(), 0, &gpu, &error), error.c_str())) return 38;
        if (!require(results_equal(view, gpu.view()), "subset CPU/GPU results differ")) return 39;
    }
    cm::clear(&source);
    return 0;
}

int test_zero_cells_and_zero_genes(bool gpu_available) {
    std::string error;
    {
        cm::compressed source;
        cs::owned_sampled_csr_structure sampled;
        cg::owned_gene_support_bitsets cpu, gpu;
        cm::init(&source);
        if (!require(fill_patterned_source(4u, 7u, &source), "failed to allocate zero-cell source")) return 40;
        if (!require(build_exact_sample(source, 0u, "support-zero-cells", &sampled, &error), error.c_str())) return 41;
        if (!require(cg::build_gene_support_bitsets_cpu(sampled.view(), &cpu, &error), error.c_str())) return 42;
        const cg::gene_support_bitset_view view = cpu.view();
        if (!require(view.layout.sampled_cell_count == 0u && view.layout.words_per_gene == 0u
                     && view.layout.support_word_count == 0u && view.gene_support == nullptr,
                     "zero-cell support layout is invalid")) return 43;
        for (std::size_t gene = 0u; gene < view.layout.gene_count; ++gene) {
            if (!require(view.detected_cell_counts[gene] == 0u, "zero-cell detection count is nonzero")) return 44;
        }
        if (gpu_available) {
            if (!require(cg::build_gene_support_bitsets_cuda(sampled.view(), 0, &gpu, &error), error.c_str())) return 45;
            if (!require(results_equal(view, gpu.view()), "zero-cell CPU/GPU results differ")) return 46;
        }
        cm::clear(&source);
    }
    {
        cm::compressed source;
        cs::sample_plan plan;
        cs::sample_spec spec;
        cs::cell_identity_view identities;
        cs::owned_sampled_csr_structure sampled;
        cg::owned_gene_support_bitsets cpu, gpu;
        cm::init(&source, 3u, 0u, 0u, cm::compressed_by_row);
        if (!require(cm::allocate(&source), "failed to allocate zero-gene source")) return 47;
        std::fill_n(source.majorPtr, 4u, 0u);
        spec.mode = cs::selection_mode::exact_lowest_hash;
        spec.seed = 9u;
        spec.split_name = "support-zero-genes";
        spec.requested_row_count = 3u;
        if (!require(cs::build_sample_plan(3u, spec, identities, &plan, &error)
                     && cs::materialize_sampled_csr_structure(&source, plan, &sampled, &error),
                     error.c_str())) return 48;
        if (!require(cg::build_gene_support_bitsets_cpu(sampled.view(), &cpu, &error), error.c_str())) return 49;
        const cg::gene_support_bitset_view view = cpu.view();
        if (!require(view.layout.sampled_cell_count == 3u && view.layout.gene_count == 0u
                     && view.layout.words_per_gene == 1u && view.layout.support_word_count == 0u
                     && view.gene_support == nullptr && view.detected_cell_counts == nullptr,
                     "zero-gene support layout is invalid")) return 50;
        if (gpu_available) {
            if (!require(cg::build_gene_support_bitsets_cuda(sampled.view(), 0, &gpu, &error), error.c_str())) return 51;
            if (!require(results_equal(view, gpu.view()), "zero-gene CPU/GPU results differ")) return 52;
        }
        cm::clear(&source);
    }
    return 0;
}

int test_invalid_indices_and_overflow(bool gpu_available) {
    const ct::ptr_t row_ptr[] = {0u, 1u};
    const ct::idx_t invalid_column[] = {2u};
    const std::uint64_t global_rows[] = {0u};
    cs::sample_provenance provenance;
    provenance.seed = 1u;
    provenance.total_rows = 1u;
    provenance.selected_rows = 1u;
    provenance.mode = cs::selection_mode::exact_lowest_hash;
    provenance.split_name = "invalid-column";
    provenance.requested_row_count = 1u;
    const cs::sampled_csr_structure_view invalid{
        1u, 2u, 1u, row_ptr, invalid_column, global_rows, &provenance
    };
    cg::owned_gene_support_bitsets output;
    std::string error;
    if (!require(!cg::build_gene_support_bitsets_cpu(invalid, &output, &error),
                 "CPU builder accepted an invalid gene index")) return 60;
    if (gpu_available) {
        error.clear();
        if (!require(!cg::build_gene_support_bitsets_cuda(invalid, 0, &output, &error),
                     "CUDA builder accepted an invalid gene index")) return 61;
    }
    cg::gene_support_layout layout;
    error.clear();
    if (!require(!cg::calculate_gene_support_layout(
                     std::numeric_limits<std::uint64_t>::max(),
                     std::numeric_limits<std::uint64_t>::max(), &layout, &error),
                 "overflowing support allocation calculation was accepted")) return 62;
    error.clear();
    if (!require(!cg::calculate_gene_support_layout(
                     0u, std::numeric_limits<std::uint64_t>::max(), &layout, &error),
                 "overflowing detection-count allocation calculation was accepted")) return 63;
    return 0;
}

int test_full_size_formula() {
    cg::gene_support_layout layout;
    std::string error;
    if (!require(cg::calculate_gene_support_layout(65536u, 30000u, &layout, &error), error.c_str())) return 70;
    if (!require(layout.words_per_gene == 2048u
                 && layout.support_word_count == 61440000u
                 && layout.support_bytes == 245760000u,
                 "65,536-cell gene-support sizing formula is incorrect")) return 71;
    return 0;
}

int full_size_gpu_smoke(bool gpu_available) {
    if (!gpu_available) {
        std::fprintf(stderr, "SKIP: full-size GPU allocation smoke (no CUDA device)\n");
        return 0;
    }
    cg::gene_support_layout layout;
    std::string error;
    if (!require(cg::calculate_gene_support_layout(65536u, 30000u, &layout, &error), error.c_str())) return 80;
    void *support = nullptr, *counts = nullptr;
    cudaError_t status = cudaMalloc(&support, layout.support_bytes);
    if (!require(status == cudaSuccess, cudaGetErrorString(status))) return 81;
    status = cudaMalloc(&counts, layout.detection_count_bytes);
    if (!require(status == cudaSuccess, cudaGetErrorString(status))) {
        cudaFree(support);
        return 82;
    }
    status = cudaMemset(support, 0, layout.support_bytes);
    if (status == cudaSuccess) status = cudaMemset(counts, 0, layout.detection_count_bytes);
    if (status == cudaSuccess) status = cudaDeviceSynchronize();
    cudaFree(counts);
    cudaFree(support);
    if (!require(status == cudaSuccess, cudaGetErrorString(status))) return 83;
    std::fprintf(stderr, "PASS: full-size GPU allocation smoke (%zu bytes support)\n", layout.support_bytes);
    return 0;
}

int main(int argc, char **argv) {
    static_assert(std::is_same<cg::support_word_t, ct::u32>::value,
                  "gene-support words must use Cellerator's canonical u32 type");
    static_assert(std::is_same<
                      decltype(std::declval<const cg::owned_gene_support_bitsets &>().sampling_provenance()),
                      const cs::sample_provenance &>::value,
                  "gene-support provenance must be exposed read-only");
    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);
    const bool gpu_available = device_status == cudaSuccess && device_count > 0;
    if (!gpu_available) {
        cudaGetLastError();
        std::fprintf(stderr, "SKIP: CUDA agreement checks (no CUDA device)\n");
    }
    int status = 0;
    if ((status = test_sample_count_boundary(31u, gpu_available)) != 0) return status;
    if ((status = test_sample_count_boundary(32u, gpu_available)) != 0) return status;
    if ((status = test_sample_count_boundary(33u, gpu_available)) != 0) return status;
    if ((status = test_deterministic_subset_alignment(gpu_available)) != 0) return status;
    if ((status = test_zero_cells_and_zero_genes(gpu_available)) != 0) return status;
    if ((status = test_invalid_indices_and_overflow(gpu_available)) != 0) return status;
    if ((status = test_full_size_formula()) != 0) return status;
    if (argc > 1 && std::strcmp(argv[1], "--full-size-gpu") == 0) {
        if ((status = full_size_gpu_smoke(gpu_available)) != 0) return status;
    }
    return 0;
}
