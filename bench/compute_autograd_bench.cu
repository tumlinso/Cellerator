#include "../src/compute/autograd/autograd.hh"
#include "benchmark_mutex.hh"
#include "cellerator_cuda_mode.hh"

#include <cuda_fp16.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace autograd = ::cellerator::compute::autograd;
namespace cs = ::cellshard;

namespace {

struct bench_config {
    std::string mode = "base-spmv";
    std::string generator = "random";
    std::uint32_t rows = 262144u;
    std::uint32_t cols = 65536u;
    std::uint32_t nnz_per_row = 64u;
    std::uint32_t out_cols = 128u;
    std::uint32_t block_size = 16u;
    std::uint32_t blocks_per_row_block = 4u;
    std::uint32_t warmup = 5u;
    std::uint32_t iters = 25u;
};

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

std::uint32_t parse_u32(const char *value, const char *label) {
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (value == nullptr || *value == '\0' || end == nullptr || *end != '\0') {
        throw std::invalid_argument(std::string("invalid integer for ") + label);
    }
    if (parsed > 0xfffffffful) throw std::out_of_range(std::string(label) + " exceeds uint32");
    return static_cast<std::uint32_t>(parsed);
}

bench_config parse_args(int argc, char **argv) {
    bench_config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        auto require_value = [&](const char *label) -> const char * {
            if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + label);
            return argv[++i];
        };
        if (arg == "--mode") {
            cfg.mode = require_value("--mode");
        } else if (arg == "--generator") {
            cfg.generator = require_value("--generator");
        } else if (arg == "--rows") {
            cfg.rows = parse_u32(require_value("--rows"), "--rows");
        } else if (arg == "--cols") {
            cfg.cols = parse_u32(require_value("--cols"), "--cols");
        } else if (arg == "--nnz-row") {
            cfg.nnz_per_row = parse_u32(require_value("--nnz-row"), "--nnz-row");
        } else if (arg == "--out-cols") {
            cfg.out_cols = parse_u32(require_value("--out-cols"), "--out-cols");
        } else if (arg == "--block-size") {
            cfg.block_size = parse_u32(require_value("--block-size"), "--block-size");
        } else if (arg == "--blocks-per-row-block") {
            cfg.blocks_per_row_block = parse_u32(require_value("--blocks-per-row-block"), "--blocks-per-row-block");
        } else if (arg == "--warmup") {
            cfg.warmup = parse_u32(require_value("--warmup"), "--warmup");
        } else if (arg == "--iters") {
            cfg.iters = parse_u32(require_value("--iters"), "--iters");
        } else if (arg == "-h" || arg == "--help") {
            std::cout
                << "Usage: computeAutogradBench [--mode base-spmv|pair-row-spmv|fleet-feature-spmv|base-csr-spmm|base-blocked-ell-spmm] "
                << "[--generator random|block-structured] [--rows N] [--cols N] [--nnz-row N] [--out-cols N] "
                << "[--block-size N] [--blocks-per-row-block N] [--warmup N] [--iters N]\n";
                std::exit(0);
        } else {
            throw std::invalid_argument(std::string("unknown argument: ") + arg);
        }
    }
    require(cfg.nnz_per_row != 0u, "nnz_per_row must be positive");
    require(cfg.cols >= cfg.nnz_per_row, "cols must be at least nnz_per_row");
    require(cfg.iters != 0u, "iters must be positive");
    require(cfg.block_size != 0u, "block_size must be positive");
    return cfg;
}

struct csr_host_matrix {
    std::unique_ptr<std::uint32_t[]> major_ptr;
    std::unique_ptr<std::uint32_t[]> minor_idx;
    std::unique_ptr<__half[]> values;
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::uint32_t nnz = 0;
};

csr_host_matrix make_host_csr(std::uint32_t rows, std::uint32_t cols, std::uint32_t nnz_per_row, std::uint32_t column_seed) {
    csr_host_matrix out;
    out.rows = rows;
    out.cols = cols;
    out.nnz = rows * nnz_per_row;
    out.major_ptr = std::make_unique<std::uint32_t[]>(static_cast<std::size_t>(rows) + 1u);
    out.minor_idx = std::make_unique<std::uint32_t[]>(out.nnz);
    out.values = std::make_unique<__half[]>(out.nnz);
    out.major_ptr[0] = 0u;
    for (std::uint32_t row = 0; row < rows; ++row) {
        const std::uint32_t base = row * nnz_per_row;
        out.major_ptr[row + 1u] = base + nnz_per_row;
        const std::uint32_t start = (row * 1315423911u + column_seed * 2654435761u) % cols;
        for (std::uint32_t j = 0; j < nnz_per_row; ++j) {
            const std::uint32_t idx = base + j;
            out.minor_idx[idx] = (start + j * 97u) % cols;
            out.values[idx] = __float2half(1.0f + static_cast<float>((idx + column_seed) % 13u) * 0.125f);
        }
    }
    return out;
}

csr_host_matrix make_host_block_sparse_csr(
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t block_size,
    std::uint32_t blocks_per_row_block,
    std::uint32_t column_seed) {
    csr_host_matrix out;
    const std::uint32_t row_blocks = (rows + block_size - 1u) / block_size;
    const std::uint32_t col_blocks = (cols + block_size - 1u) / block_size;

    require((rows % block_size) == 0u, "block-structured generator requires rows divisible by block_size");
    require((cols % block_size) == 0u, "block-structured generator requires cols divisible by block_size");
    require(blocks_per_row_block != 0u, "blocks_per_row_block must be positive");
    require(col_blocks >= blocks_per_row_block, "cols too small for requested block pattern");

    out.rows = rows;
    out.cols = cols;
    out.nnz = rows * block_size * blocks_per_row_block;
    out.major_ptr = std::make_unique<std::uint32_t[]>(static_cast<std::size_t>(rows) + 1u);
    out.minor_idx = std::make_unique<std::uint32_t[]>(out.nnz);
    out.values = std::make_unique<__half[]>(out.nnz);
    out.major_ptr[0] = 0u;

    for (std::uint32_t rb = 0u; rb < row_blocks; ++rb) {
        std::unique_ptr<std::uint32_t[]> block_cols(new std::uint32_t[blocks_per_row_block]);
        const std::uint32_t start = (rb * 2654435761u + column_seed * 40503u) % col_blocks;
        for (std::uint32_t slot = 0u; slot < blocks_per_row_block; ++slot) {
            block_cols[slot] = (start + slot * 3u) % col_blocks;
        }
        for (std::uint32_t r_in = 0u; r_in < block_size; ++r_in) {
            const std::uint32_t row = rb * block_size + r_in;
            const std::uint32_t base = row * block_size * blocks_per_row_block;
            out.major_ptr[row + 1u] = base + block_size * blocks_per_row_block;
            std::uint32_t cursor = base;
            for (std::uint32_t slot = 0u; slot < blocks_per_row_block; ++slot) {
                const std::uint32_t col_block = block_cols[slot];
                for (std::uint32_t c_in = 0u; c_in < block_size; ++c_in) {
                    out.minor_idx[cursor] = col_block * block_size + c_in;
                    out.values[cursor] = __float2half(0.5f + static_cast<float>((cursor + column_seed) % 17u) * 0.0625f);
                    ++cursor;
                }
            }
        }
    }
    return out;
}

csr_host_matrix make_host_matrix_for_cfg(const bench_config &cfg, std::uint32_t column_seed) {
    if (cfg.generator == "block-structured") {
        return make_host_block_sparse_csr(cfg.rows, cfg.cols, cfg.block_size, cfg.blocks_per_row_block, column_seed);
    }
    return make_host_csr(cfg.rows, cfg.cols, cfg.nnz_per_row, column_seed);
}

std::unique_ptr<float[]> make_host_vector(std::uint32_t count, std::uint32_t seed) {
    auto out = std::make_unique<float[]>(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        out[i] = 0.25f + static_cast<float>((i + seed) % 31u) * 0.03125f;
    }
    return out;
}

std::unique_ptr<float[]> make_host_matrix(std::uint32_t rows, std::uint32_t cols, std::uint32_t seed) {
    auto out = std::make_unique<float[]>(static_cast<std::size_t>(rows) * cols);
    for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t c = 0; c < cols; ++c) {
            out[static_cast<std::size_t>(r) * cols + c] = 0.125f + static_cast<float>((r * 17u + c + seed) % 29u) * 0.0625f;
        }
    }
    return out;
}

std::unique_ptr<__half[]> make_host_half_matrix(std::uint32_t rows, std::uint32_t cols, std::uint32_t seed) {
    auto out = std::make_unique<__half[]>(static_cast<std::size_t>(rows) * cols);
    for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t c = 0; c < cols; ++c) {
            out[static_cast<std::size_t>(r) * cols + c] = __float2half(0.125f + static_cast<float>((r * 17u + c + seed) % 29u) * 0.0625f);
        }
    }
    return out;
}

template<typename T>
autograd::device_buffer<T> allocate_on_device(int device, std::size_t count) {
    autograd::cuda_require(cudaSetDevice(device), "cudaSetDevice(allocate_on_device)");
    return autograd::allocate_device_buffer<T>(count);
}

template<typename T>
void upload_on_device(int device, autograd::device_buffer<T> *dst, const T *src, std::size_t count) {
    autograd::cuda_require(cudaSetDevice(device), "cudaSetDevice(upload_on_device)");
    autograd::upload_device_buffer(dst, src, count);
}

double elapsed_ms(const std::chrono::steady_clock::time_point &start, const std::chrono::steady_clock::time_point &stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

void print_summary(const bench_config &cfg, double total_ms, std::uint64_t total_nnz, std::uint64_t output_elements, std::uint32_t device_count_used) {
    const double avg_ms = total_ms / static_cast<double>(cfg.iters);
    const double nnz_per_s = static_cast<double>(total_nnz) * static_cast<double>(cfg.iters) / (total_ms * 1.0e-3);
    const double out_per_s = static_cast<double>(output_elements) * static_cast<double>(cfg.iters) / (total_ms * 1.0e-3);
    std::cout
        << "mode=" << cfg.mode
        << " generator=" << cfg.generator
        << " devices=" << device_count_used
        << " rows=" << cfg.rows
        << " cols=" << cfg.cols
        << " nnz_per_row=" << cfg.nnz_per_row
        << " out_cols=" << cfg.out_cols
        << " block_size=" << cfg.block_size
        << " blocks_per_row_block=" << cfg.blocks_per_row_block
        << " warmup=" << cfg.warmup
        << " iters=" << cfg.iters
        << " avg_ms=" << avg_ms
        << " nnz_per_s=" << nnz_per_s
        << " out_el_per_s=" << out_per_s
        << '\n';
}

void print_blocked_ell_meta(const cs::sparse::blocked_ell &blocked, double convert_ms) {
    const double logical = static_cast<double>(blocked.nnz);
    const double resident = static_cast<double>(blocked.rows) * blocked.ell_cols;
    const double fill = resident > 0.0 ? logical / resident : 1.0;
    std::cout
        << "blocked_ell"
        << " convert_ms=" << convert_ms
        << " block_size=" << blocked.block_size
        << " ell_cols=" << blocked.ell_cols
        << " ell_width_blocks=" << cs::sparse::ell_width_blocks(&blocked)
        << " fill_ratio=" << fill
        << '\n';
}

void run_base_spmv(const bench_config &cfg) {
    autograd::execution_context ctx;
    autograd::init(&ctx, 0);

    const csr_host_matrix matrix = make_host_matrix_for_cfg(cfg, 0u);
    const auto vector = make_host_vector(cfg.cols, 0u);

    auto major_ptr = autograd::allocate_device_buffer<std::uint32_t>(static_cast<std::size_t>(cfg.rows) + 1u);
    auto minor_idx = autograd::allocate_device_buffer<std::uint32_t>(matrix.nnz);
    auto values = autograd::allocate_device_buffer<__half>(matrix.nnz);
    auto dense = autograd::allocate_device_buffer<float>(cfg.cols);
    auto out = autograd::allocate_device_buffer<float>(cfg.rows);

    autograd::upload_device_buffer(&major_ptr, matrix.major_ptr.get(), static_cast<std::size_t>(cfg.rows) + 1u);
    autograd::upload_device_buffer(&minor_idx, matrix.minor_idx.get(), matrix.nnz);
    autograd::upload_device_buffer(&values, matrix.values.get(), matrix.nnz);
    autograd::upload_device_buffer(&dense, vector.get(), cfg.cols);
    autograd::cuda_require(cudaStreamSynchronize(ctx.stream), "cudaStreamSynchronize(base setup)");

    for (std::uint32_t i = 0; i < cfg.warmup; ++i) {
        autograd::base::csr_spmv_fwd_f16_f32(ctx, major_ptr.data, minor_idx.data, values.data, cfg.rows, dense.data, out.data);
    }
    autograd::cuda_require(cudaStreamSynchronize(ctx.stream), "cudaStreamSynchronize(base warmup)");

    const auto start = std::chrono::steady_clock::now();
    for (std::uint32_t i = 0; i < cfg.iters; ++i) {
        autograd::base::csr_spmv_fwd_f16_f32(ctx, major_ptr.data, minor_idx.data, values.data, cfg.rows, dense.data, out.data);
    }
    autograd::cuda_require(cudaStreamSynchronize(ctx.stream), "cudaStreamSynchronize(base timed)");
    const auto stop = std::chrono::steady_clock::now();

    print_summary(cfg, elapsed_ms(start, stop), matrix.nnz, cfg.rows, 1u);
    autograd::clear(&ctx);
}

void run_base_csr_spmm(const bench_config &cfg) {
    autograd::execution_context ctx;
    autograd::init(&ctx, 0);

    const csr_host_matrix matrix = make_host_matrix_for_cfg(cfg, 0u);
    const auto rhs = make_host_matrix(cfg.cols, cfg.out_cols, 7u);

    auto major_ptr = autograd::allocate_device_buffer<std::uint32_t>(static_cast<std::size_t>(cfg.rows) + 1u);
    auto minor_idx = autograd::allocate_device_buffer<std::uint32_t>(matrix.nnz);
    auto values = autograd::allocate_device_buffer<__half>(matrix.nnz);
    auto dense = autograd::allocate_device_buffer<float>(static_cast<std::size_t>(cfg.cols) * cfg.out_cols);
    auto out = autograd::allocate_device_buffer<float>(static_cast<std::size_t>(cfg.rows) * cfg.out_cols);

    autograd::upload_device_buffer(&major_ptr, matrix.major_ptr.get(), static_cast<std::size_t>(cfg.rows) + 1u);
    autograd::upload_device_buffer(&minor_idx, matrix.minor_idx.get(), matrix.nnz);
    autograd::upload_device_buffer(&values, matrix.values.get(), matrix.nnz);
    autograd::upload_device_buffer(&dense, rhs.get(), static_cast<std::size_t>(cfg.cols) * cfg.out_cols);
    autograd::cuda_require(cudaStreamSynchronize(ctx.stream), "cudaStreamSynchronize(csr spmm setup)");

    for (std::uint32_t i = 0; i < cfg.warmup; ++i) {
        autograd::base::csr_spmm_fwd_f16_f32(ctx, major_ptr.data, minor_idx.data, values.data, cfg.rows, cfg.cols, dense.data, cfg.out_cols, cfg.out_cols, out.data, cfg.out_cols);
    }
    autograd::cuda_require(cudaStreamSynchronize(ctx.stream), "cudaStreamSynchronize(csr spmm warmup)");

    const auto start = std::chrono::steady_clock::now();
    for (std::uint32_t i = 0; i < cfg.iters; ++i) {
        autograd::base::csr_spmm_fwd_f16_f32(ctx, major_ptr.data, minor_idx.data, values.data, cfg.rows, cfg.cols, dense.data, cfg.out_cols, cfg.out_cols, out.data, cfg.out_cols);
    }
    autograd::cuda_require(cudaStreamSynchronize(ctx.stream), "cudaStreamSynchronize(csr spmm timed)");
    const auto stop = std::chrono::steady_clock::now();

    print_summary(cfg, elapsed_ms(start, stop), matrix.nnz, static_cast<std::uint64_t>(cfg.rows) * cfg.out_cols, 1u);
    autograd::clear(&ctx);
}

void run_base_blocked_ell_spmm(const bench_config &cfg) {
    autograd::execution_context ctx;
    autograd::init(&ctx, 0);

    const csr_host_matrix matrix = make_host_matrix_for_cfg(cfg, 0u);
    const auto rhs = make_host_half_matrix(cfg.cols, cfg.out_cols, 7u);
    const unsigned int candidates[] = { cfg.block_size, 8u, 16u, 32u };
    cs::convert::blocked_ell_tune_result tune = {};
    cs::sparse::compressed host_csr;
    cs::sparse::blocked_ell blocked;
    const auto convert_start = std::chrono::steady_clock::now();
    cs::sparse::init(&host_csr, cfg.rows, cfg.cols, matrix.nnz, cs::sparse::compressed_by_row);
    cs::sparse::init(&blocked);
    host_csr.majorPtr = matrix.major_ptr.get();
    host_csr.minorIdx = matrix.minor_idx.get();
    host_csr.val = matrix.values.get();
    if (cfg.generator == "block-structured") {
        require(cs::convert::blocked_ell_from_compressed(&host_csr, cfg.block_size, &blocked) != 0, "blocked ell fixed conversion failed");
        tune.block_size = cfg.block_size;
    } else {
        require(cs::convert::blocked_ell_from_compressed_auto(&host_csr, candidates, 4u, &blocked, &tune) != 0, "blocked ell conversion failed");
    }
    const auto convert_stop = std::chrono::steady_clock::now();

    auto block_col_idx = autograd::allocate_device_buffer<std::uint32_t>(static_cast<std::size_t>(cs::sparse::row_block_count(&blocked)) * cs::sparse::ell_width_blocks(&blocked));
    auto blocked_values = autograd::allocate_device_buffer<__half>(static_cast<std::size_t>(blocked.rows) * blocked.ell_cols);
    auto dense = autograd::allocate_device_buffer<__half>(static_cast<std::size_t>(cfg.cols) * cfg.out_cols);
    auto out = autograd::allocate_device_buffer<float>(static_cast<std::size_t>(cfg.rows) * cfg.out_cols);
    autograd::cusparse_cache cache;
    autograd::init(&cache);

    autograd::upload_device_buffer(&block_col_idx, blocked.blockColIdx, static_cast<std::size_t>(cs::sparse::row_block_count(&blocked)) * cs::sparse::ell_width_blocks(&blocked));
    autograd::upload_device_buffer(&blocked_values, blocked.val, static_cast<std::size_t>(blocked.rows) * blocked.ell_cols);
    autograd::upload_device_buffer(&dense, rhs.get(), static_cast<std::size_t>(cfg.cols) * cfg.out_cols);
    autograd::cuda_require(cudaStreamSynchronize(ctx.stream), "cudaStreamSynchronize(blocked ell spmm setup)");
    print_blocked_ell_meta(blocked, elapsed_ms(convert_start, convert_stop));

    for (std::uint32_t i = 0; i < cfg.warmup; ++i) {
        autograd::base::blocked_ell_spmm_fwd_f16_f16_f32_lib(ctx, &cache, blocked.val, block_col_idx.data, blocked_values.data, blocked.rows, blocked.cols, blocked.block_size, blocked.ell_cols, dense.data, cfg.out_cols, cfg.out_cols, out.data, cfg.out_cols);
    }
    autograd::cuda_require(cudaStreamSynchronize(ctx.stream), "cudaStreamSynchronize(blocked ell spmm warmup)");

    const auto start = std::chrono::steady_clock::now();
    for (std::uint32_t i = 0; i < cfg.iters; ++i) {
        autograd::base::blocked_ell_spmm_fwd_f16_f16_f32_lib(ctx, &cache, blocked.val, block_col_idx.data, blocked_values.data, blocked.rows, blocked.cols, blocked.block_size, blocked.ell_cols, dense.data, cfg.out_cols, cfg.out_cols, out.data, cfg.out_cols);
    }
    autograd::cuda_require(cudaStreamSynchronize(ctx.stream), "cudaStreamSynchronize(blocked ell spmm timed)");
    const auto stop = std::chrono::steady_clock::now();

    print_summary(cfg, elapsed_ms(start, stop), matrix.nnz, static_cast<std::uint64_t>(cfg.rows) * cfg.out_cols, 1u);
    autograd::clear(&cache);
    cs::sparse::clear(&blocked);
    autograd::clear(&ctx);
}

void run_pair_row_spmv(const bench_config &cfg) {
    autograd::fleet_context fleet;
    autograd::init(&fleet);
    autograd::discover_fleet(&fleet, true, cudaStreamNonBlocking, true);
    require(fleet.local.device_count >= 3u, "pair-row-spmv requires visible slots 0 and 2");

    const unsigned int slots[2] = { 0u, 2u };
    const std::uint32_t rows0 = cfg.rows / 2u;
    const std::uint32_t rows1 = cfg.rows - rows0;
    bench_config cfg0 = cfg;
    bench_config cfg1 = cfg;
    cfg0.rows = rows0;
    cfg1.rows = rows1;
    const csr_host_matrix matrix0 = make_host_matrix_for_cfg(cfg0, 0u);
    const csr_host_matrix matrix1 = make_host_matrix_for_cfg(cfg1, 17u);
    const auto vector = make_host_vector(cfg.cols, 0u);

    const int device0 = autograd::fleet_device_id(fleet, slots[0]);
    const int device1 = autograd::fleet_device_id(fleet, slots[1]);

    auto major0 = allocate_on_device<std::uint32_t>(device0, static_cast<std::size_t>(rows0) + 1u);
    auto minor0 = allocate_on_device<std::uint32_t>(device0, matrix0.nnz);
    auto values0 = allocate_on_device<__half>(device0, matrix0.nnz);
    auto vector0 = allocate_on_device<float>(device0, cfg.cols);
    auto out0 = allocate_on_device<float>(device0, rows0);

    auto major1 = allocate_on_device<std::uint32_t>(device1, static_cast<std::size_t>(rows1) + 1u);
    auto minor1 = allocate_on_device<std::uint32_t>(device1, matrix1.nnz);
    auto values1 = allocate_on_device<__half>(device1, matrix1.nnz);
    auto vector1 = allocate_on_device<float>(device1, cfg.cols);
    auto out1 = allocate_on_device<float>(device1, rows1);

    upload_on_device(device0, &major0, matrix0.major_ptr.get(), static_cast<std::size_t>(rows0) + 1u);
    upload_on_device(device0, &minor0, matrix0.minor_idx.get(), matrix0.nnz);
    upload_on_device(device0, &values0, matrix0.values.get(), matrix0.nnz);
    upload_on_device(device0, &vector0, vector.get(), cfg.cols);

    upload_on_device(device1, &major1, matrix1.major_ptr.get(), static_cast<std::size_t>(rows1) + 1u);
    upload_on_device(device1, &minor1, matrix1.minor_idx.get(), matrix1.nnz);
    upload_on_device(device1, &values1, matrix1.values.get(), matrix1.nnz);
    upload_on_device(device1, &vector1, vector.get(), cfg.cols);

    const std::uint32_t *major_ptrs[] = { major0.data, major1.data };
    const std::uint32_t *minor_ptrs[] = { minor0.data, minor1.data };
    const __half *value_ptrs[] = { values0.data, values1.data };
    const float *vector_ptrs[] = { vector0.data, vector1.data };
    const std::uint32_t rows[] = { rows0, rows1 };
    float *out_ptrs[] = { out0.data, out1.data };

    autograd::synchronize_slots(fleet, slots, 2u);
    for (std::uint32_t i = 0; i < cfg.warmup; ++i) {
        autograd::dist::launch_csr_spmv_fwd_f16_f32(&fleet, slots, 2u, major_ptrs, minor_ptrs, value_ptrs, rows, vector_ptrs, out_ptrs);
    }
    autograd::synchronize_slots(fleet, slots, 2u);

    const auto start = std::chrono::steady_clock::now();
    for (std::uint32_t i = 0; i < cfg.iters; ++i) {
        autograd::dist::launch_csr_spmv_fwd_f16_f32(&fleet, slots, 2u, major_ptrs, minor_ptrs, value_ptrs, rows, vector_ptrs, out_ptrs);
    }
    autograd::synchronize_slots(fleet, slots, 2u);
    const auto stop = std::chrono::steady_clock::now();

    print_summary(cfg, elapsed_ms(start, stop), static_cast<std::uint64_t>(matrix0.nnz) + matrix1.nnz, cfg.rows, 2u);
    autograd::clear(&fleet);
}

void run_fleet_feature_spmv(const bench_config &cfg) {
    autograd::fleet_context fleet;
    autograd::init(&fleet);
    autograd::discover_fleet(&fleet, true, cudaStreamNonBlocking, true);
    require(fleet.local.device_count >= 4u, "fleet-feature-spmv requires 4 visible GPUs");
    require((cfg.cols % 4u) == 0u, "fleet-feature-spmv requires cols divisible by 4");

    unsigned int slots[4] = {};
    require(autograd::default_mode_fleet_slots(fleet, slots, 4u) == 4u, "default fleet slots unavailable");
    const std::uint32_t cols_per_slot = cfg.cols / 4u;

    bench_config shard_cfg = cfg;
    shard_cfg.cols = cols_per_slot;
    csr_host_matrix matrices[4] = {
        make_host_matrix_for_cfg(shard_cfg, 0u),
        make_host_matrix_for_cfg(shard_cfg, 11u),
        make_host_matrix_for_cfg(shard_cfg, 23u),
        make_host_matrix_for_cfg(shard_cfg, 37u)
    };
    auto vectors0 = make_host_vector(cols_per_slot, 0u);
    auto vectors1 = make_host_vector(cols_per_slot, 11u);
    auto vectors2 = make_host_vector(cols_per_slot, 23u);
    auto vectors3 = make_host_vector(cols_per_slot, 37u);

    auto major0 = allocate_on_device<std::uint32_t>(autograd::fleet_device_id(fleet, slots[0]), static_cast<std::size_t>(cfg.rows) + 1u);
    auto minor0 = allocate_on_device<std::uint32_t>(autograd::fleet_device_id(fleet, slots[0]), matrices[0].nnz);
    auto values0 = allocate_on_device<__half>(autograd::fleet_device_id(fleet, slots[0]), matrices[0].nnz);
    auto vector0 = allocate_on_device<float>(autograd::fleet_device_id(fleet, slots[0]), cols_per_slot);
    auto out0 = allocate_on_device<float>(autograd::fleet_device_id(fleet, slots[0]), cfg.rows);
    auto reduced0 = allocate_on_device<float>(autograd::fleet_device_id(fleet, slots[0]), cfg.rows);

    auto major1 = allocate_on_device<std::uint32_t>(autograd::fleet_device_id(fleet, slots[1]), static_cast<std::size_t>(cfg.rows) + 1u);
    auto minor1 = allocate_on_device<std::uint32_t>(autograd::fleet_device_id(fleet, slots[1]), matrices[1].nnz);
    auto values1 = allocate_on_device<__half>(autograd::fleet_device_id(fleet, slots[1]), matrices[1].nnz);
    auto vector1 = allocate_on_device<float>(autograd::fleet_device_id(fleet, slots[1]), cols_per_slot);
    auto out1 = allocate_on_device<float>(autograd::fleet_device_id(fleet, slots[1]), cfg.rows);

    auto major2 = allocate_on_device<std::uint32_t>(autograd::fleet_device_id(fleet, slots[2]), static_cast<std::size_t>(cfg.rows) + 1u);
    auto minor2 = allocate_on_device<std::uint32_t>(autograd::fleet_device_id(fleet, slots[2]), matrices[2].nnz);
    auto values2 = allocate_on_device<__half>(autograd::fleet_device_id(fleet, slots[2]), matrices[2].nnz);
    auto vector2 = allocate_on_device<float>(autograd::fleet_device_id(fleet, slots[2]), cols_per_slot);
    auto out2 = allocate_on_device<float>(autograd::fleet_device_id(fleet, slots[2]), cfg.rows);

    auto major3 = allocate_on_device<std::uint32_t>(autograd::fleet_device_id(fleet, slots[3]), static_cast<std::size_t>(cfg.rows) + 1u);
    auto minor3 = allocate_on_device<std::uint32_t>(autograd::fleet_device_id(fleet, slots[3]), matrices[3].nnz);
    auto values3 = allocate_on_device<__half>(autograd::fleet_device_id(fleet, slots[3]), matrices[3].nnz);
    auto vector3 = allocate_on_device<float>(autograd::fleet_device_id(fleet, slots[3]), cols_per_slot);
    auto out3 = allocate_on_device<float>(autograd::fleet_device_id(fleet, slots[3]), cfg.rows);

    upload_on_device(autograd::fleet_device_id(fleet, slots[0]), &major0, matrices[0].major_ptr.get(), static_cast<std::size_t>(cfg.rows) + 1u);
    upload_on_device(autograd::fleet_device_id(fleet, slots[0]), &minor0, matrices[0].minor_idx.get(), matrices[0].nnz);
    upload_on_device(autograd::fleet_device_id(fleet, slots[0]), &values0, matrices[0].values.get(), matrices[0].nnz);
    upload_on_device(autograd::fleet_device_id(fleet, slots[0]), &vector0, vectors0.get(), cols_per_slot);

    upload_on_device(autograd::fleet_device_id(fleet, slots[1]), &major1, matrices[1].major_ptr.get(), static_cast<std::size_t>(cfg.rows) + 1u);
    upload_on_device(autograd::fleet_device_id(fleet, slots[1]), &minor1, matrices[1].minor_idx.get(), matrices[1].nnz);
    upload_on_device(autograd::fleet_device_id(fleet, slots[1]), &values1, matrices[1].values.get(), matrices[1].nnz);
    upload_on_device(autograd::fleet_device_id(fleet, slots[1]), &vector1, vectors1.get(), cols_per_slot);

    upload_on_device(autograd::fleet_device_id(fleet, slots[2]), &major2, matrices[2].major_ptr.get(), static_cast<std::size_t>(cfg.rows) + 1u);
    upload_on_device(autograd::fleet_device_id(fleet, slots[2]), &minor2, matrices[2].minor_idx.get(), matrices[2].nnz);
    upload_on_device(autograd::fleet_device_id(fleet, slots[2]), &values2, matrices[2].values.get(), matrices[2].nnz);
    upload_on_device(autograd::fleet_device_id(fleet, slots[2]), &vector2, vectors2.get(), cols_per_slot);

    upload_on_device(autograd::fleet_device_id(fleet, slots[3]), &major3, matrices[3].major_ptr.get(), static_cast<std::size_t>(cfg.rows) + 1u);
    upload_on_device(autograd::fleet_device_id(fleet, slots[3]), &minor3, matrices[3].minor_idx.get(), matrices[3].nnz);
    upload_on_device(autograd::fleet_device_id(fleet, slots[3]), &values3, matrices[3].values.get(), matrices[3].nnz);
    upload_on_device(autograd::fleet_device_id(fleet, slots[3]), &vector3, vectors3.get(), cols_per_slot);

    const std::uint32_t *major_ptrs[] = { major0.data, major1.data, major2.data, major3.data };
    const std::uint32_t *minor_ptrs[] = { minor0.data, minor1.data, minor2.data, minor3.data };
    const __half *value_ptrs[] = { values0.data, values1.data, values2.data, values3.data };
    const float *vector_ptrs[] = { vector0.data, vector1.data, vector2.data, vector3.data };
    const std::uint32_t rows[] = { cfg.rows, cfg.rows, cfg.rows, cfg.rows };
    float *out_ptrs[] = { out0.data, out1.data, out2.data, out3.data };
    const float *partial_ptrs[] = { out0.data, out1.data, out2.data, out3.data };

    autograd::synchronize_slots(fleet, slots, 4u);
    for (std::uint32_t i = 0; i < cfg.warmup; ++i) {
        autograd::dist::launch_csr_spmv_fwd_f16_f32(&fleet, slots, 4u, major_ptrs, minor_ptrs, value_ptrs, rows, vector_ptrs, out_ptrs);
        autograd::dist::reduce_sum_to_leader_f32(&fleet, slots, 4u, partial_ptrs, cfg.rows, reduced0.data);
    }
    const unsigned int leader_only[1] = { slots[0] };
    autograd::synchronize_slots(fleet, leader_only, 1u);

    const auto start = std::chrono::steady_clock::now();
    for (std::uint32_t i = 0; i < cfg.iters; ++i) {
        autograd::dist::launch_csr_spmv_fwd_f16_f32(&fleet, slots, 4u, major_ptrs, minor_ptrs, value_ptrs, rows, vector_ptrs, out_ptrs);
        autograd::dist::reduce_sum_to_leader_f32(&fleet, slots, 4u, partial_ptrs, cfg.rows, reduced0.data);
    }
    autograd::synchronize_slots(fleet, leader_only, 1u);
    const auto stop = std::chrono::steady_clock::now();

    std::uint64_t total_nnz = 0;
    for (const auto &matrix : matrices) total_nnz += matrix.nnz;
    print_summary(cfg, elapsed_ms(start, stop), total_nnz, cfg.rows, 4u);
    autograd::clear(&fleet);
}

} // namespace

int main(int argc, char **argv) {
    cellerator::bench::benchmark_mutex_guard benchmark_mutex("computeAutogradBench");
    const bench_config cfg = parse_args(argc, argv);
    std::cout << "cuda_mode=" << cellerator::build::cuda_mode_name << '\n';
    if (cfg.mode == "base-spmv") {
        run_base_spmv(cfg);
        return 0;
    }
    if (cfg.mode == "base-csr-spmm") {
        run_base_csr_spmm(cfg);
        return 0;
    }
    if (cfg.mode == "base-blocked-ell-spmm") {
        run_base_blocked_ell_spmm(cfg);
        return 0;
    }
    if (cfg.mode == "pair-row-spmv") {
        run_pair_row_spmv(cfg);
        return 0;
    }
    if (cfg.mode == "fleet-feature-spmv") {
        run_fleet_feature_spmv(cfg);
        return 0;
    }
    throw std::invalid_argument("unknown bench mode");
}
