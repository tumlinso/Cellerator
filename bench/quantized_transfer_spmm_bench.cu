#include <Cellerator/compute/autograd.hh>
#include <Cellerator/quantized/api.cuh>
#include "../extern/CellShard/src/convert/blocked_ell_from_compressed.cuh"
#include "benchmark_mutex.hh"

#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace autograd = ::cellerator::compute::autograd;
namespace cs = ::cellshard;
namespace msq = ::cellerator::quantized;

namespace {

struct bench_config {
    std::string generator = "block-structured";
    std::string policy = "per_gene_affine";
    std::uint32_t rows = 65536u;
    std::uint32_t cols = 8192u;
    std::uint32_t nnz_per_row = 64u;
    std::uint32_t out_cols = 128u;
    std::uint32_t block_size = 16u;
    std::uint32_t blocks_per_row_block = 4u;
    std::uint32_t bits = 4u;
    std::uint32_t warmup = 5u;
    std::uint32_t iters = 25u;
};

struct csr_host_matrix {
    std::unique_ptr<std::uint32_t[]> major_ptr;
    std::unique_ptr<std::uint32_t[]> minor_idx;
    std::unique_ptr<__half[]> values;
    std::uint32_t rows = 0u;
    std::uint32_t cols = 0u;
    std::uint32_t nnz = 0u;
};

struct quantized_host_payload {
    std::vector<std::uint8_t> packed_values;
    std::vector<float> column_scales;
    std::vector<float> column_offsets;
    std::vector<float> row_offsets;
    std::uint32_t row_stride_bytes = 0u;
    std::uint32_t decode_policy = msq::blocked_ell::decode_policy_unknown;
};

struct bench_result {
    std::string case_name;
    std::string policy;
    std::uint32_t bits = 0u;
    std::uint64_t payload_bytes = 0u;
    std::uint64_t h2d_bytes = 0u;
    double host_prep_ms = 0.0;
    double upload_ms = 0.0;
    double kernel_ms = 0.0;
    double total_ms = 0.0;
    double mean_abs_err = 0.0;
    double max_abs_err = 0.0;
    std::vector<float> output;
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

void print_usage(const char *argv0) {
    std::fprintf(stderr,
                 "Usage: %s [options]\n"
                 "  --generator random|block-structured   Default: block-structured\n"
                 "  --policy per_gene_affine|column_scale_row_offset   Default: per_gene_affine\n"
                 "  --rows N                              Default: 65536\n"
                 "  --cols N                              Default: 8192\n"
                 "  --nnz-row N                           Default: 64\n"
                 "  --out-cols N                          Default: 128\n"
                 "  --block-size N                        Default: 16\n"
                 "  --blocks-per-row-block N              Default: 4\n"
                 "  --bits {1|2|4|8}                      Default: 4\n"
                 "  --warmup N                            Default: 5\n"
                 "  --iters N                             Default: 25\n",
                 argv0);
}

bench_config parse_args(int argc, char **argv) {
    bench_config cfg;

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        auto require_value = [&](const char *label) -> const char * {
            if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + label);
            return argv[++i];
        };

        if (arg == "--generator") {
            cfg.generator = require_value("--generator");
        } else if (arg == "--policy") {
            cfg.policy = require_value("--policy");
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
        } else if (arg == "--bits") {
            cfg.bits = parse_u32(require_value("--bits"), "--bits");
        } else if (arg == "--warmup") {
            cfg.warmup = parse_u32(require_value("--warmup"), "--warmup");
        } else if (arg == "--iters") {
            cfg.iters = parse_u32(require_value("--iters"), "--iters");
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument(std::string("unknown argument: ") + arg);
        }
    }

    require(cfg.rows != 0u, "rows must be positive");
    require(cfg.cols != 0u, "cols must be positive");
    require(cfg.out_cols != 0u, "out_cols must be positive");
    require(cfg.block_size != 0u, "block_size must be positive");
    require(cfg.iters != 0u, "iters must be positive");
    require(msq::valid_bits(static_cast<int>(cfg.bits)), "bits must be 1, 2, 4, or 8");
    require(cfg.policy == "per_gene_affine" || cfg.policy == "column_scale_row_offset",
            "policy must be per_gene_affine or column_scale_row_offset");
    require(cfg.generator == "random" || cfg.generator == "block-structured",
            "generator must be random or block-structured");
    if (cfg.generator == "block-structured") {
        require((cfg.rows % cfg.block_size) == 0u, "block-structured generator requires rows divisible by block_size");
        require((cfg.cols % cfg.block_size) == 0u, "block-structured generator requires cols divisible by block_size");
        require(cfg.blocks_per_row_block != 0u, "blocks_per_row_block must be positive");
    } else {
        require(cfg.cols >= cfg.nnz_per_row, "cols must be at least nnz_per_row");
    }
    return cfg;
}

csr_host_matrix make_host_csr(std::uint32_t rows, std::uint32_t cols, std::uint32_t nnz_per_row, std::uint32_t column_seed) {
    csr_host_matrix out;

    out.rows = rows;
    out.cols = cols;
    out.nnz = rows * nnz_per_row;
    out.major_ptr = std::make_unique<std::uint32_t[]>(static_cast<std::size_t>(rows) + 1u);
    out.minor_idx = std::make_unique<std::uint32_t[]>(out.nnz);
    out.values = std::make_unique<__half[]>(out.nnz);
    out.major_ptr[0] = 0u;
    for (std::uint32_t row = 0u; row < rows; ++row) {
        const std::uint32_t base = row * nnz_per_row;
        out.major_ptr[row + 1u] = base + nnz_per_row;
        const std::uint32_t start = (row * 1315423911u + column_seed * 2654435761u) % cols;
        for (std::uint32_t j = 0u; j < nnz_per_row; ++j) {
            const std::uint32_t idx = base + j;
            out.minor_idx[idx] = (start + j * 97u) % cols;
            out.values[idx] = __float2half(0.5f + static_cast<float>((idx + column_seed) % 17u) * 0.0625f);
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

    require(col_blocks >= blocks_per_row_block, "cols too small for requested block pattern");
    out.rows = rows;
    out.cols = cols;
    out.nnz = rows * block_size * blocks_per_row_block;
    out.major_ptr = std::make_unique<std::uint32_t[]>(static_cast<std::size_t>(rows) + 1u);
    out.minor_idx = std::make_unique<std::uint32_t[]>(out.nnz);
    out.values = std::make_unique<__half[]>(out.nnz);
    out.major_ptr[0] = 0u;

    for (std::uint32_t rb = 0u; rb < row_blocks; ++rb) {
        std::vector<std::uint32_t> block_cols(blocks_per_row_block, 0u);
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

std::unique_ptr<float[]> make_host_rhs(std::uint32_t rows, std::uint32_t cols, std::uint32_t seed) {
    auto out = std::make_unique<float[]>(static_cast<std::size_t>(rows) * cols);
    for (std::uint32_t r = 0u; r < rows; ++r) {
        for (std::uint32_t c = 0u; c < cols; ++c) {
            out[static_cast<std::size_t>(r) * cols + c] =
                0.125f + static_cast<float>((r * 17u + c + seed) % 29u) * 0.0625f;
        }
    }
    return out;
}

double elapsed_ms(const std::chrono::steady_clock::time_point &start, const std::chrono::steady_clock::time_point &stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

std::uint32_t column_for_slot(const cs::sparse::blocked_ell &blocked, std::uint32_t row, std::uint32_t slot) {
    const std::uint32_t width = cs::sparse::ell_width_blocks(&blocked);
    const std::uint32_t row_block = row / blocked.block_size;
    const std::uint32_t slot_block = slot / blocked.block_size;
    const std::uint32_t slot_offset = slot % blocked.block_size;
    const std::uint32_t block_col = blocked.blockColIdx[static_cast<std::size_t>(row_block) * width + slot_block];

    if (block_col == cs::sparse::blocked_ell_invalid_col) return cs::sparse::blocked_ell_invalid_col;
    return block_col * blocked.block_size + slot_offset;
}

void extract_slot_values(
    const cs::sparse::blocked_ell &blocked,
    std::vector<float> *slot_values,
    std::vector<int> *slot_columns) {
    const std::size_t total_slots = static_cast<std::size_t>(blocked.rows) * blocked.ell_cols;

    slot_values->assign(total_slots, 0.0f);
    slot_columns->assign(total_slots, -1);
    for (std::uint32_t row = 0u; row < blocked.rows; ++row) {
        for (std::uint32_t slot = 0u; slot < blocked.ell_cols; ++slot) {
            const std::size_t idx = static_cast<std::size_t>(row) * blocked.ell_cols + slot;
            const std::uint32_t col = column_for_slot(blocked, row, slot);
            if (col >= blocked.cols) continue;
            (*slot_values)[idx] = __half2float(blocked.val[idx]);
            (*slot_columns)[idx] = static_cast<int>(col);
        }
    }
}

void calibrate_per_gene_affine(
    const cs::sparse::blocked_ell &blocked,
    const std::vector<float> &slot_values,
    const std::vector<int> &slot_columns,
    std::uint32_t max_code,
    std::vector<float> *column_scales,
    std::vector<float> *column_offsets) {
    std::vector<float> mins(blocked.cols, std::numeric_limits<float>::infinity());
    std::vector<float> maxs(blocked.cols, -std::numeric_limits<float>::infinity());
    std::vector<std::uint8_t> seen(blocked.cols, 0u);

    column_scales->assign(blocked.cols, 1.0f);
    column_offsets->assign(blocked.cols, 0.0f);

    for (std::size_t i = 0; i < slot_columns.size(); ++i) {
        const int col = slot_columns[i];
        if (col < 0) continue;
        mins[static_cast<std::size_t>(col)] = std::min(mins[static_cast<std::size_t>(col)], slot_values[i]);
        maxs[static_cast<std::size_t>(col)] = std::max(maxs[static_cast<std::size_t>(col)], slot_values[i]);
        seen[static_cast<std::size_t>(col)] = 1u;
    }

    for (std::uint32_t col = 0u; col < blocked.cols; ++col) {
        if (seen[col] == 0u) continue;
        const float lo = mins[col];
        const float hi = maxs[col];
        const float scale = (hi > lo && max_code != 0u)
            ? (hi - lo) / static_cast<float>(max_code)
            : 1.0f;
        (*column_scales)[col] = scale > 1.0e-6f ? scale : 1.0e-6f;
        (*column_offsets)[col] = lo;
    }
}

void calibrate_column_scale_row_offset(
    const cs::sparse::blocked_ell &blocked,
    const std::vector<float> &slot_values,
    const std::vector<int> &slot_columns,
    std::uint32_t max_code,
    std::vector<float> *column_scales,
    std::vector<float> *row_offsets) {
    std::vector<float> column_max_delta(blocked.cols, 0.0f);

    column_scales->assign(blocked.cols, 1.0f);
    row_offsets->assign(blocked.rows, 0.0f);

    for (std::uint32_t row = 0u; row < blocked.rows; ++row) {
        float row_min = std::numeric_limits<float>::infinity();
        bool seen = false;
        for (std::uint32_t slot = 0u; slot < blocked.ell_cols; ++slot) {
            const std::size_t idx = static_cast<std::size_t>(row) * blocked.ell_cols + slot;
            if (slot_columns[idx] < 0) continue;
            row_min = std::min(row_min, slot_values[idx]);
            seen = true;
        }
        (*row_offsets)[row] = seen ? row_min : 0.0f;
    }

    for (std::uint32_t row = 0u; row < blocked.rows; ++row) {
        for (std::uint32_t slot = 0u; slot < blocked.ell_cols; ++slot) {
            const std::size_t idx = static_cast<std::size_t>(row) * blocked.ell_cols + slot;
            const int col = slot_columns[idx];
            if (col < 0) continue;
            const float delta = slot_values[idx] - (*row_offsets)[row];
            column_max_delta[static_cast<std::size_t>(col)] =
                std::max(column_max_delta[static_cast<std::size_t>(col)], delta);
        }
    }

    for (std::uint32_t col = 0u; col < blocked.cols; ++col) {
        const float scale = (column_max_delta[col] > 0.0f && max_code != 0u)
            ? column_max_delta[col] / static_cast<float>(max_code)
            : 1.0f;
        (*column_scales)[col] = scale > 1.0e-6f ? scale : 1.0e-6f;
    }
}

template<int Bits>
bool pack_quantized_payload_impl(
    const cs::sparse::blocked_ell &blocked,
    const std::vector<float> &slot_values,
    const bench_config &cfg,
    quantized_host_payload *out) {
    out->row_stride_bytes = static_cast<std::uint32_t>(msq::blocked_ell::aligned_row_bytes<Bits>(static_cast<int>(blocked.ell_cols)));
    out->packed_values.assign(static_cast<std::size_t>(blocked.rows) * out->row_stride_bytes, 0u);

    if (cfg.policy == "per_gene_affine") {
        out->decode_policy = msq::blocked_ell::decode_policy_per_gene_affine;
        auto matrix = msq::blocked_ell::make_matrix<Bits>(
            static_cast<int>(blocked.rows),
            static_cast<int>(blocked.cols),
            static_cast<int>(blocked.nnz),
            static_cast<int>(blocked.block_size),
            static_cast<int>(blocked.ell_cols),
            static_cast<int>(out->row_stride_bytes),
            blocked.blockColIdx,
            out->packed_values.data(),
            msq::make_per_gene_affine(out->column_scales.data(), out->column_offsets.data()));
        return msq::blocked_ell::pack_row_major_values(&matrix, slot_values.data()) == 0;
    }

    out->decode_policy = msq::blocked_ell::decode_policy_column_scale_row_offset;
    auto matrix = msq::blocked_ell::make_matrix<Bits>(
        static_cast<int>(blocked.rows),
        static_cast<int>(blocked.cols),
        static_cast<int>(blocked.nnz),
        static_cast<int>(blocked.block_size),
        static_cast<int>(blocked.ell_cols),
        static_cast<int>(out->row_stride_bytes),
        blocked.blockColIdx,
        out->packed_values.data(),
        msq::make_column_scale_row_offset(out->column_scales.data(), out->row_offsets.data()));
    return msq::blocked_ell::pack_row_major_values(&matrix, slot_values.data()) == 0;
}

quantized_host_payload build_quantized_payload(
    const cs::sparse::blocked_ell &blocked,
    const std::vector<float> &slot_values,
    const std::vector<int> &slot_columns,
    const bench_config &cfg,
    double *host_prep_ms) {
    quantized_host_payload out;
    const auto start = std::chrono::steady_clock::now();
    const std::uint32_t max_code = (1u << cfg.bits) - 1u;

    if (cfg.policy == "per_gene_affine") {
        calibrate_per_gene_affine(blocked, slot_values, slot_columns, max_code, &out.column_scales, &out.column_offsets);
    } else {
        calibrate_column_scale_row_offset(blocked, slot_values, slot_columns, max_code, &out.column_scales, &out.row_offsets);
    }

    const bool ok = msq::dispatch_bits(static_cast<int>(cfg.bits), [&](auto bit_tag) {
        constexpr int kBits = decltype(bit_tag)::value;
        return pack_quantized_payload_impl<kBits>(blocked, slot_values, cfg, &out);
    });
    require(ok, "quantized host pack failed");
    const auto stop = std::chrono::steady_clock::now();
    *host_prep_ms = elapsed_ms(start, stop);
    return out;
}

double compute_mean_abs_error(const std::vector<float> &lhs, const std::vector<float> &rhs) {
    double sum = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        sum += std::fabs(static_cast<double>(lhs[i]) - static_cast<double>(rhs[i]));
    }
    return lhs.empty() ? 0.0 : sum / static_cast<double>(lhs.size());
}

double compute_max_abs_error(const std::vector<float> &lhs, const std::vector<float> &rhs) {
    double max_err = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        max_err = std::max(max_err, std::fabs(static_cast<double>(lhs[i]) - static_cast<double>(rhs[i])));
    }
    return max_err;
}

bench_result run_blocked_ell_case(
    const bench_config &cfg,
    const cs::sparse::blocked_ell &blocked,
    const float *rhs_host) {
    bench_result out;
    autograd::execution_context ctx;
    autograd::cusparse_cache cache;

    out.case_name = "blocked_ell_f16";
    out.policy = "native";
    autograd::init(&ctx, 0);
    autograd::init(&cache);

    auto block_col_idx = autograd::allocate_device_buffer<std::uint32_t>(
        static_cast<std::size_t>(cs::sparse::row_block_count(&blocked)) * cs::sparse::ell_width_blocks(&blocked));
    auto blocked_values = autograd::allocate_device_buffer<__half>(static_cast<std::size_t>(blocked.rows) * blocked.ell_cols);
    auto rhs = autograd::allocate_device_buffer<float>(static_cast<std::size_t>(blocked.cols) * cfg.out_cols);
    auto out_dev = autograd::allocate_device_buffer<float>(static_cast<std::size_t>(blocked.rows) * cfg.out_cols);

    const auto upload_start = std::chrono::steady_clock::now();
    autograd::upload_device_buffer(&block_col_idx,
                                   blocked.blockColIdx,
                                   static_cast<std::size_t>(cs::sparse::row_block_count(&blocked)) * cs::sparse::ell_width_blocks(&blocked));
    autograd::upload_device_buffer(&blocked_values, blocked.val, static_cast<std::size_t>(blocked.rows) * blocked.ell_cols);
    autograd::upload_device_buffer(&rhs, rhs_host, static_cast<std::size_t>(blocked.cols) * cfg.out_cols);
    autograd::cuda_require(cudaStreamSynchronize(ctx.stream), "cudaStreamSynchronize(blocked upload)");
    const auto upload_stop = std::chrono::steady_clock::now();

    for (std::uint32_t i = 0u; i < cfg.warmup; ++i) {
        autograd::base::blocked_ell_spmm_fwd_f16_f32_lib(ctx,
                                                         &cache,
                                                         blocked.val,
                                                         block_col_idx.data,
                                                         blocked_values.data,
                                                         blocked.rows,
                                                         blocked.cols,
                                                         blocked.block_size,
                                                         blocked.ell_cols,
                                                         rhs.data,
                                                         cfg.out_cols,
                                                         cfg.out_cols,
                                                         out_dev.data,
                                                         cfg.out_cols);
    }
    autograd::cuda_require(cudaStreamSynchronize(ctx.stream), "cudaStreamSynchronize(blocked warmup)");

    const auto kernel_start = std::chrono::steady_clock::now();
    for (std::uint32_t i = 0u; i < cfg.iters; ++i) {
        autograd::base::blocked_ell_spmm_fwd_f16_f32_lib(ctx,
                                                         &cache,
                                                         blocked.val,
                                                         block_col_idx.data,
                                                         blocked_values.data,
                                                         blocked.rows,
                                                         blocked.cols,
                                                         blocked.block_size,
                                                         blocked.ell_cols,
                                                         rhs.data,
                                                         cfg.out_cols,
                                                         cfg.out_cols,
                                                         out_dev.data,
                                                         cfg.out_cols);
    }
    autograd::cuda_require(cudaStreamSynchronize(ctx.stream), "cudaStreamSynchronize(blocked timed)");
    const auto kernel_stop = std::chrono::steady_clock::now();

    out.output.resize(static_cast<std::size_t>(blocked.rows) * cfg.out_cols, 0.0f);
    autograd::download_device_buffer(out_dev, out.output.data(), out.output.size());

    out.payload_bytes =
        static_cast<std::uint64_t>(cs::sparse::row_block_count(&blocked))
            * static_cast<std::uint64_t>(cs::sparse::ell_width_blocks(&blocked))
            * sizeof(std::uint32_t)
        + static_cast<std::uint64_t>(blocked.rows) * static_cast<std::uint64_t>(blocked.ell_cols) * sizeof(__half);
    out.h2d_bytes = out.payload_bytes + static_cast<std::uint64_t>(blocked.cols) * cfg.out_cols * sizeof(float);
    out.upload_ms = elapsed_ms(upload_start, upload_stop);
    out.kernel_ms = elapsed_ms(kernel_start, kernel_stop);
    out.total_ms = out.upload_ms + out.kernel_ms;

    autograd::clear(&cache);
    autograd::clear(&ctx);
    return out;
}

bench_result run_quantized_case(
    const bench_config &cfg,
    const cs::sparse::blocked_ell &blocked,
    const quantized_host_payload &payload,
    const float *rhs_host,
    const std::vector<float> &reference_output) {
    bench_result out;
    autograd::execution_context ctx;

    out.case_name = "quantized_blocked_ell";
    out.policy = cfg.policy;
    out.bits = cfg.bits;
    autograd::init(&ctx, 0);

    auto block_col_idx = autograd::allocate_device_buffer<std::uint32_t>(
        static_cast<std::size_t>(cs::sparse::row_block_count(&blocked)) * cs::sparse::ell_width_blocks(&blocked));
    auto packed_values = autograd::allocate_device_buffer<std::uint8_t>(payload.packed_values.size());
    auto column_scales = autograd::allocate_device_buffer<float>(payload.column_scales.size());
    auto rhs = autograd::allocate_device_buffer<float>(static_cast<std::size_t>(blocked.cols) * cfg.out_cols);
    auto out_dev = autograd::allocate_device_buffer<float>(static_cast<std::size_t>(blocked.rows) * cfg.out_cols);
    autograd::device_buffer<float> column_offsets;
    autograd::device_buffer<float> row_offsets;

    if (!payload.column_offsets.empty()) {
        column_offsets = autograd::allocate_device_buffer<float>(payload.column_offsets.size());
    }
    if (!payload.row_offsets.empty()) {
        row_offsets = autograd::allocate_device_buffer<float>(payload.row_offsets.size());
    }

    const auto upload_start = std::chrono::steady_clock::now();
    autograd::upload_device_buffer(&block_col_idx,
                                   blocked.blockColIdx,
                                   static_cast<std::size_t>(cs::sparse::row_block_count(&blocked)) * cs::sparse::ell_width_blocks(&blocked));
    autograd::upload_device_buffer(&packed_values, payload.packed_values.data(), payload.packed_values.size());
    autograd::upload_device_buffer(&column_scales, payload.column_scales.data(), payload.column_scales.size());
    if (!payload.column_offsets.empty()) {
        autograd::upload_device_buffer(&column_offsets, payload.column_offsets.data(), payload.column_offsets.size());
    }
    if (!payload.row_offsets.empty()) {
        autograd::upload_device_buffer(&row_offsets, payload.row_offsets.data(), payload.row_offsets.size());
    }
    autograd::upload_device_buffer(&rhs, rhs_host, static_cast<std::size_t>(blocked.cols) * cfg.out_cols);
    autograd::cuda_require(cudaStreamSynchronize(ctx.stream), "cudaStreamSynchronize(quantized upload)");
    const auto upload_stop = std::chrono::steady_clock::now();

    autograd::quantized_blocked_ell_view view{};
    view.rows = blocked.rows;
    view.cols = blocked.cols;
    view.nnz = blocked.nnz;
    view.block_size = blocked.block_size;
    view.ell_cols = blocked.ell_cols;
    view.row_stride_bytes = payload.row_stride_bytes;
    view.bits = cfg.bits;
    view.decode_policy = payload.decode_policy;
    view.block_col_idx = block_col_idx.data;
    view.packed_values = packed_values.data;
    view.column_scales = column_scales.data;
    view.column_offsets = payload.column_offsets.empty() ? nullptr : column_offsets.data;
    view.row_offsets = payload.row_offsets.empty() ? nullptr : row_offsets.data;

    for (std::uint32_t i = 0u; i < cfg.warmup; ++i) {
        autograd::base::quantized_blocked_ell_spmm_fwd_f32(ctx, view, rhs.data, cfg.out_cols, cfg.out_cols, out_dev.data, cfg.out_cols);
    }
    autograd::cuda_require(cudaStreamSynchronize(ctx.stream), "cudaStreamSynchronize(quantized warmup)");

    const auto kernel_start = std::chrono::steady_clock::now();
    for (std::uint32_t i = 0u; i < cfg.iters; ++i) {
        autograd::base::quantized_blocked_ell_spmm_fwd_f32(ctx, view, rhs.data, cfg.out_cols, cfg.out_cols, out_dev.data, cfg.out_cols);
    }
    autograd::cuda_require(cudaStreamSynchronize(ctx.stream), "cudaStreamSynchronize(quantized timed)");
    const auto kernel_stop = std::chrono::steady_clock::now();

    out.output.resize(static_cast<std::size_t>(blocked.rows) * cfg.out_cols, 0.0f);
    autograd::download_device_buffer(out_dev, out.output.data(), out.output.size());

    out.payload_bytes =
        static_cast<std::uint64_t>(cs::sparse::row_block_count(&blocked))
            * static_cast<std::uint64_t>(cs::sparse::ell_width_blocks(&blocked))
            * sizeof(std::uint32_t)
        + static_cast<std::uint64_t>(payload.packed_values.size()) * sizeof(std::uint8_t)
        + static_cast<std::uint64_t>(payload.column_scales.size()) * sizeof(float)
        + static_cast<std::uint64_t>(payload.column_offsets.size()) * sizeof(float)
        + static_cast<std::uint64_t>(payload.row_offsets.size()) * sizeof(float);
    out.h2d_bytes = out.payload_bytes + static_cast<std::uint64_t>(blocked.cols) * cfg.out_cols * sizeof(float);
    out.upload_ms = elapsed_ms(upload_start, upload_stop);
    out.kernel_ms = elapsed_ms(kernel_start, kernel_stop);
    out.total_ms = out.upload_ms + out.kernel_ms;
    out.mean_abs_err = compute_mean_abs_error(out.output, reference_output);
    out.max_abs_err = compute_max_abs_error(out.output, reference_output);

    autograd::clear(&ctx);
    return out;
}

void print_case(
    const bench_config &cfg,
    const bench_result &result,
    const bench_result *reference) {
    const double upload_ratio = reference != nullptr && result.upload_ms > 0.0 ? reference->upload_ms / result.upload_ms : 1.0;
    const double kernel_ratio = reference != nullptr && result.kernel_ms > 0.0 ? reference->kernel_ms / result.kernel_ms : 1.0;
    const double total_ratio = reference != nullptr && result.total_ms > 0.0 ? reference->total_ms / result.total_ms : 1.0;
    const double payload_ratio = reference != nullptr && result.payload_bytes != 0u
        ? static_cast<double>(reference->payload_bytes) / static_cast<double>(result.payload_bytes)
        : 1.0;

    std::cout
        << "case=" << result.case_name
        << " generator=" << cfg.generator
        << " policy=" << result.policy
        << " bits=" << result.bits
        << " rows=" << cfg.rows
        << " cols=" << cfg.cols
        << " out_cols=" << cfg.out_cols
        << " block_size=" << cfg.block_size
        << " payload_bytes=" << result.payload_bytes
        << " h2d_bytes=" << result.h2d_bytes
        << " host_prep_ms=" << result.host_prep_ms
        << " upload_ms=" << result.upload_ms
        << " kernel_ms=" << result.kernel_ms
        << " total_ms=" << result.total_ms
        << " mean_abs_err=" << result.mean_abs_err
        << " max_abs_err=" << result.max_abs_err
        << " payload_ratio_vs_blocked=" << payload_ratio
        << " upload_speedup_vs_blocked=" << upload_ratio
        << " kernel_speedup_vs_blocked=" << kernel_ratio
        << " total_speedup_vs_blocked=" << total_ratio
        << '\n';
}

} // namespace

int main(int argc, char **argv) {
    int device_count = 0;
    bench_config cfg = parse_args(argc, argv);
    const auto rhs = make_host_rhs(cfg.cols, cfg.out_cols, 7u);
    const csr_host_matrix matrix = make_host_matrix_for_cfg(cfg, 0u);
    cs::sparse::compressed host_csr;
    cs::sparse::blocked_ell blocked;
    std::vector<float> slot_values;
    std::vector<int> slot_columns;
    double quantized_host_prep_ms = 0.0;

    autograd::cuda_require(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    require(device_count > 0, "quantizedTransferSpmmBench requires at least one visible CUDA device");
    cellerator::bench::benchmark_mutex_guard benchmark_mutex("quantizedTransferSpmmBench");

    cs::sparse::init(&host_csr, cfg.rows, cfg.cols, matrix.nnz, cs::sparse::compressed_by_row);
    cs::sparse::init(&blocked);
    host_csr.majorPtr = matrix.major_ptr.get();
    host_csr.minorIdx = matrix.minor_idx.get();
    host_csr.val = matrix.values.get();
    require(cs::convert::blocked_ell_from_compressed(&host_csr, cfg.block_size, &blocked) != 0,
            "blocked_ell_from_compressed failed");

    extract_slot_values(blocked, &slot_values, &slot_columns);
    quantized_host_payload quantized = build_quantized_payload(blocked, slot_values, slot_columns, cfg, &quantized_host_prep_ms);

    bench_result blocked_result = run_blocked_ell_case(cfg, blocked, rhs.get());
    bench_result quantized_result = run_quantized_case(cfg, blocked, quantized, rhs.get(), blocked_result.output);
    quantized_result.host_prep_ms = quantized_host_prep_ms;

    print_case(cfg, blocked_result, &blocked_result);
    print_case(cfg, quantized_result, &blocked_result);

    cs::sparse::clear(&blocked);
    return 0;
}
