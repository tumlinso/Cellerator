#include "benchmark_mutex.hh"
#include <Cellerator/compute/autograd.hh>
#include "../extern/CellShard/src/convert/blocked_ell_from_compressed.cuh"
#include <Cellerator/ingest/compressed_parts.cuh>
#include <Cellerator/ingest/dataset_ingest.cuh>
#include <Cellerator/ingest/mtx_reader.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace autograd = ::cellerator::compute::autograd;
namespace cdataset = ::cellerator::ingest::dataset;
namespace cmtx = ::cellerator::ingest::mtx;
namespace cs = ::cellshard;

namespace {

struct real_case {
    std::string dataset_id;
    std::string matrix_path;
    std::uint32_t rows = 0u;
    std::uint32_t cols = 0u;
    std::uint64_t nnz = 0u;
    std::uint32_t slice_rows = 0u;
    std::uint32_t rhs_cols = 256u;
};

struct config {
    std::string manifest_path = "bench/real_data/generated/blocked_ell_optimization/manifest.tsv";
    std::string dataset_filter = "all";
    std::string algorithm_filter = "all";
    std::string json_path;
    unsigned int device = 0u;
    unsigned int parts = 4u;
    unsigned int reader_bytes_mb = 16u;
    unsigned int block_size = 16u;
    unsigned int bucket_cap = 16u;
    unsigned int warmup = 2u;
    unsigned int iters = 5u;
    unsigned int local_search_passes = 3u;
    unsigned int rhs_cols = 128u;
    int run_spmm = 1;
};

struct compressed_owned {
    cs::sparse::compressed *part = nullptr;

    ~compressed_owned() {
        if (part != nullptr) {
            cs::sparse::clear(part);
            delete part;
        }
    }

    compressed_owned() = default;
    compressed_owned(const compressed_owned &) = delete;
    compressed_owned &operator=(const compressed_owned &) = delete;

    compressed_owned(compressed_owned &&other) noexcept : part(other.part) {
        other.part = nullptr;
    }

    compressed_owned &operator=(compressed_owned &&other) noexcept {
        if (this == &other) return *this;
        if (part != nullptr) {
            cs::sparse::clear(part);
            delete part;
        }
        part = other.part;
        other.part = nullptr;
        return *this;
    }
};

struct blocked_ell_owned {
    cs::sparse::blocked_ell *part = nullptr;

    ~blocked_ell_owned() {
        if (part != nullptr) {
            cs::sparse::clear(part);
            delete part;
        }
    }

    blocked_ell_owned() = default;
    blocked_ell_owned(const blocked_ell_owned &) = delete;
    blocked_ell_owned &operator=(const blocked_ell_owned &) = delete;

    blocked_ell_owned(blocked_ell_owned &&other) noexcept : part(other.part) {
        other.part = nullptr;
    }

    blocked_ell_owned &operator=(blocked_ell_owned &&other) noexcept {
        if (this == &other) return *this;
        if (part != nullptr) {
            cs::sparse::clear(part);
            delete part;
        }
        part = other.part;
        other.part = nullptr;
        return *this;
    }
};

struct case_matrices {
    std::vector<compressed_owned> compressed_parts;
    std::vector<blocked_ell_owned> blocked_parts;
    std::uint32_t rows = 0u;
    std::uint32_t cols = 0u;
    std::uint64_t nnz = 0u;
};

struct part_descriptor {
    const cs::sparse::blocked_ell *part = nullptr;
    std::uint32_t row_block_begin = 0u;
    std::uint32_t row_block_count = 0u;
};

struct support_model {
    std::uint32_t cols = 0u;
    std::uint32_t total_row_blocks = 0u;
    std::uint32_t word_count = 0u;
    std::uint32_t block_size = 0u;
    std::vector<part_descriptor> parts;
    std::vector<std::uint64_t> column_bits;
    std::vector<std::uint32_t> support_count;
    std::vector<std::uint32_t> first_row_block;
};

struct width_eval {
    std::uint64_t objective_bytes = 0u;
    std::uint64_t execution_payload_bytes = 0u;
    std::uint64_t segment_value_bytes = 0u;
    std::uint64_t segment_index_bytes = 0u;
    std::uint32_t total_segments = 0u;
    std::uint32_t max_bucket_count = 0u;
    double weighted_fill_ratio = 0.0;
    std::vector<std::uint32_t> chosen_bucket_counts;
};

struct algorithm_result {
    std::string algorithm;
    std::string dataset_id;
    std::uint32_t rows = 0u;
    std::uint32_t cols = 0u;
    std::uint64_t nnz = 0u;
    std::uint32_t block_size = 0u;
    std::uint64_t csr_bytes = 0u;
    std::uint64_t canonical_blocked_bytes = 0u;
    std::uint64_t optimized_shard_bytes = 0u;
    std::uint64_t execution_payload_bytes = 0u;
    std::uint64_t segment_value_bytes = 0u;
    std::uint64_t segment_index_bytes = 0u;
    std::uint32_t total_segments = 0u;
    std::uint32_t max_bucket_count = 0u;
    double weighted_fill_ratio = 0.0;
    double optimized_vs_csr = 0.0;
    double canonical_vs_csr = 0.0;
    double build_ms = 0.0;
    double spmm_ms = 0.0;
};

struct dp_layout {
    std::vector<std::uint32_t> row_block_order;
    std::vector<std::uint32_t> segment_row_block_offsets;
    std::vector<std::uint32_t> segment_width_blocks;
    std::uint64_t total_bytes = 0u;
};

struct device_segment {
    autograd::device_buffer<std::uint32_t> block_col_idx;
    autograd::device_buffer<__half> values;
    autograd::device_buffer<float> out;
    std::uint32_t rows = 0u;
    std::uint32_t cols = 0u;
    std::uint32_t block_size = 0u;
    std::uint32_t ell_cols = 0u;
};

struct partition_device_segments {
    std::vector<device_segment> segments;
};

static void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

static std::uint32_t parse_u32(const char *text, const char *label) {
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (text == nullptr || *text == '\0' || end == nullptr || *end != '\0' || parsed > 0xfffffffful) {
        throw std::invalid_argument(std::string("invalid integer for ") + label);
    }
    return static_cast<std::uint32_t>(parsed);
}

static void usage(const char *argv0) {
    std::fprintf(stderr,
                 "Usage: %s [options]\n"
                 "  --manifest PATH         Real-data manifest. Default: bench/real_data/generated/blocked_ell_optimization/manifest.tsv\n"
                 "  --dataset ID|all        Dataset filter. Default: all\n"
                 "  --algorithm NAME|all    baseline | exact-dp | overlap | local-search | all\n"
                 "  --device N              CUDA device for SpMM. Default: 0\n"
                 "  --parts N               Part count. Default: 4\n"
                 "  --reader-bytes-mb N     MTX reader buffer in MiB. Default: 16\n"
                 "  --block-size N          Fixed blocked-ELL block size. Default: 16\n"
                 "  --bucket-cap N          Maximum exact-DP buckets per part. Default: 16\n"
                 "  --warmup N              SpMM warmup iterations. Default: 2\n"
                 "  --iters N               SpMM timed iterations. Default: 5\n"
                 "  --rhs-cols N            Dense RHS columns for SpMM. Default: 128\n"
                 "  --local-search-passes N Local-search passes. Default: 3\n"
                 "  --json-out PATH         Optional JSON summary output\n"
                 "  --skip-spmm             Skip CUDA SpMM runtime proxy\n",
                 argv0);
}

static config parse_args(int argc, char **argv) {
    config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        auto require_value = [&](const char *label) -> const char * {
            if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + label);
            return argv[++i];
        };

        if (arg == "--manifest") {
            cfg.manifest_path = require_value("--manifest");
        } else if (arg == "--dataset") {
            cfg.dataset_filter = require_value("--dataset");
        } else if (arg == "--algorithm") {
            cfg.algorithm_filter = require_value("--algorithm");
        } else if (arg == "--device") {
            cfg.device = parse_u32(require_value("--device"), "--device");
        } else if (arg == "--parts") {
            cfg.parts = parse_u32(require_value("--parts"), "--parts");
        } else if (arg == "--reader-bytes-mb") {
            cfg.reader_bytes_mb = parse_u32(require_value("--reader-bytes-mb"), "--reader-bytes-mb");
        } else if (arg == "--block-size") {
            cfg.block_size = parse_u32(require_value("--block-size"), "--block-size");
        } else if (arg == "--bucket-cap") {
            cfg.bucket_cap = parse_u32(require_value("--bucket-cap"), "--bucket-cap");
        } else if (arg == "--warmup") {
            cfg.warmup = parse_u32(require_value("--warmup"), "--warmup");
        } else if (arg == "--iters") {
            cfg.iters = parse_u32(require_value("--iters"), "--iters");
        } else if (arg == "--rhs-cols") {
            cfg.rhs_cols = parse_u32(require_value("--rhs-cols"), "--rhs-cols");
        } else if (arg == "--local-search-passes") {
            cfg.local_search_passes = parse_u32(require_value("--local-search-passes"), "--local-search-passes");
        } else if (arg == "--json-out") {
            cfg.json_path = require_value("--json-out");
        } else if (arg == "--skip-spmm") {
            cfg.run_spmm = 0;
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument(std::string("unknown argument: ") + arg);
        }
    }

    if (cfg.parts == 0u) throw std::invalid_argument("--parts must be non-zero");
    if (cfg.block_size == 0u) throw std::invalid_argument("--block-size must be non-zero");
    if (cfg.bucket_cap == 0u) throw std::invalid_argument("--bucket-cap must be non-zero");
    if (cfg.iters == 0u) throw std::invalid_argument("--iters must be non-zero");
    return cfg;
}

static std::vector<std::string> split_tab(const std::string &line) {
    std::vector<std::string> out;
    std::size_t begin = 0u;
    while (begin <= line.size()) {
        const std::size_t end = line.find('\t', begin);
        if (end == std::string::npos) {
            out.push_back(line.substr(begin));
            break;
        }
        out.push_back(line.substr(begin, end - begin));
        begin = end + 1u;
    }
    return out;
}

static std::vector<real_case> load_manifest(const std::string &path, const std::string &filter) {
    std::ifstream in(path);
    std::string line;
    std::vector<real_case> out;

    if (!in) throw std::runtime_error("failed to open manifest");
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string> fields = split_tab(line);
        if (fields.size() < 7u) throw std::runtime_error("manifest requires 7 tab-separated columns");
        if (fields[0] == "dataset_id") continue;
        real_case entry;
        entry.dataset_id = fields[0];
        entry.matrix_path = fields[1];
        entry.rows = static_cast<std::uint32_t>(std::stoul(fields[2]));
        entry.cols = static_cast<std::uint32_t>(std::stoul(fields[3]));
        entry.nnz = static_cast<std::uint64_t>(std::stoull(fields[4]));
        entry.slice_rows = static_cast<std::uint32_t>(std::stoul(fields[5]));
        entry.rhs_cols = static_cast<std::uint32_t>(std::stoul(fields[6]));
        if (filter != "all" && filter != entry.dataset_id) continue;
        out.push_back(std::move(entry));
    }
    if (out.empty()) throw std::runtime_error("manifest produced no cases");
    return out;
}

static std::vector<unsigned long> build_balanced_offsets(const unsigned long *row_nnz,
                                                         unsigned long slice_rows,
                                                         unsigned int shard_count,
                                                         unsigned long total_rows,
                                                         unsigned long row_alignment) {
    std::vector<unsigned long> offsets(static_cast<std::size_t>(shard_count) + 2u, 0ul);
    unsigned long total_nnz = 0ul;
    unsigned long running = 0ul;
    unsigned int next_cut = 1u;

    for (unsigned long row = 0; row < slice_rows; ++row) total_nnz += row_nnz[row];
    for (unsigned long row = 0; row < slice_rows && next_cut < shard_count; ++row) {
        running += row_nnz[row];
        const unsigned long target =
            (total_nnz * static_cast<unsigned long>(next_cut) + static_cast<unsigned long>(shard_count) - 1ul)
            / static_cast<unsigned long>(shard_count);
        if (running >= target) {
            unsigned long cut = row + 1ul;
            if (row_alignment > 1ul) {
                cut = (cut / row_alignment) * row_alignment;
                if (cut == 0ul) cut = row_alignment;
                if (cut > slice_rows) cut = slice_rows;
            }
            if (cut <= offsets[next_cut - 1u]) {
                cut = offsets[next_cut - 1u] + row_alignment;
                if (cut > slice_rows) cut = slice_rows;
            }
            offsets[next_cut] = cut;
            ++next_cut;
        }
    }
    while (next_cut < shard_count) {
        offsets[next_cut] = slice_rows;
        ++next_cut;
    }
    offsets[shard_count] = slice_rows;
    offsets[shard_count + 1u] = total_rows;
    return offsets;
}

static std::vector<compressed_owned> convert_window_parts_to_csr(const cs::sharded<cs::sparse::coo> &coo_view,
                                                                 cmtx::compressed_workspace *ws) {
    std::vector<compressed_owned> out;
    out.reserve(coo_view.num_partitions);
    for (unsigned long part = 0; part < coo_view.num_partitions; ++part) {
        compressed_owned owned;
        owned.part = new cs::sparse::compressed();
        cs::sparse::init(owned.part);
        if (!cmtx::reserve(ws,
                           static_cast<unsigned int>(coo_view.parts[part]->rows),
                           static_cast<unsigned int>(coo_view.parts[part]->cols),
                           static_cast<unsigned int>(coo_view.parts[part]->nnz))) {
            throw std::runtime_error("compressed workspace reserve failed");
        }
        if (coo_view.parts[part]->nnz != 0u) {
            std::memcpy(ws->h_row_idx,
                        coo_view.parts[part]->rowIdx,
                        static_cast<std::size_t>(coo_view.parts[part]->nnz) * sizeof(unsigned int));
            std::memcpy(ws->h_col_idx,
                        coo_view.parts[part]->colIdx,
                        static_cast<std::size_t>(coo_view.parts[part]->nnz) * sizeof(unsigned int));
            std::memcpy(ws->h_in_val,
                        coo_view.parts[part]->val,
                        static_cast<std::size_t>(coo_view.parts[part]->nnz) * sizeof(__half));
        }
        if (!cmtx::build_pinned_triplet_to_compressed(ws,
                                                      static_cast<unsigned int>(coo_view.parts[part]->rows),
                                                      static_cast<unsigned int>(coo_view.parts[part]->cols),
                                                      static_cast<unsigned int>(coo_view.parts[part]->nnz),
                                                      cs::sparse::compressed_by_row)) {
            throw std::runtime_error("build_pinned_triplet_to_compressed failed");
        }
        cs::sparse::clear(owned.part);
        cs::sparse::init(owned.part,
                         static_cast<cs::types::dim_t>(coo_view.parts[part]->rows),
                         static_cast<cs::types::dim_t>(coo_view.parts[part]->cols),
                         static_cast<cs::types::nnz_t>(coo_view.parts[part]->nnz),
                         cs::sparse::compressed_by_row);
        if (!cs::sparse::allocate(owned.part)) {
            throw std::runtime_error("compressed allocate failed");
        }
        for (unsigned int row = 0u; row <= static_cast<unsigned int>(coo_view.parts[part]->rows); ++row) {
            owned.part->majorPtr[row] = static_cast<cs::types::ptr_t>(ws->h_major_ptr[row]);
        }
        if (coo_view.parts[part]->nnz != 0u) {
            std::memcpy(owned.part->minorIdx,
                        ws->h_minor_idx,
                        static_cast<std::size_t>(coo_view.parts[part]->nnz) * sizeof(unsigned int));
            std::memcpy(owned.part->val,
                        ws->h_out_val,
                        static_cast<std::size_t>(coo_view.parts[part]->nnz) * sizeof(__half));
        }
        if (owned.part == nullptr) {
            throw std::runtime_error("convert_coo_part_to_csr failed");
        }
        out.push_back(std::move(owned));
    }
    return out;
}

static std::vector<blocked_ell_owned> convert_window_parts_to_blocked_ell(const std::vector<compressed_owned> &compressed_parts,
                                                                          std::uint32_t block_size) {
    std::vector<blocked_ell_owned> out;
    out.reserve(compressed_parts.size());
    for (const compressed_owned &compressed : compressed_parts) {
        blocked_ell_owned owned;
        owned.part = new cs::sparse::blocked_ell();
        cs::sparse::init(owned.part);
        if (!cs::convert::blocked_ell_from_compressed(compressed.part, block_size, owned.part)) {
            throw std::runtime_error("blocked_ell_from_compressed failed");
        }
        out.push_back(std::move(owned));
    }
    return out;
}

static case_matrices load_case_matrices(const real_case &entry, const config &cfg) {
    cmtx::header header{};
    std::unique_ptr<unsigned long[]> row_nnz;
    cs::sharded<cs::sparse::coo> coo_view{};
    unsigned long *part_nnz = nullptr;
    cmtx::compressed_workspace workspace{};
    case_matrices out;

    cs::init(&coo_view);
    cmtx::init(&workspace);
    require(cmtx::setup(&workspace, static_cast<int>(cfg.device), static_cast<cudaStream_t>(0)),
            "compressed workspace setup failed");
    if (!cmtx::scan_row_nnz(entry.matrix_path.c_str(),
                            &header,
                            reinterpret_cast<unsigned long **>(&row_nnz),
                            static_cast<std::size_t>(cfg.reader_bytes_mb) << 20u)) {
        throw std::runtime_error("scan_row_nnz failed");
    }
    const std::vector<unsigned long> offsets =
        build_balanced_offsets(row_nnz.get(), entry.slice_rows, cfg.parts, header.rows, static_cast<unsigned long>(cfg.block_size));
    if (!cmtx::count_all_part_nnz(entry.matrix_path.c_str(),
                                  &header,
                                  offsets.data(),
                                  cfg.parts + 1u,
                                  &part_nnz)) {
        cmtx::clear(&workspace);
        throw std::runtime_error("count_all_part_nnz failed");
    }
    if (!cmtx::load_part_window_coo(entry.matrix_path.c_str(),
                                    &header,
                                    offsets.data(),
                                    part_nnz,
                                    cfg.parts + 1u,
                                    0ul,
                                    cfg.parts,
                                    &coo_view,
                                    static_cast<std::size_t>(cfg.reader_bytes_mb) << 20u)) {
        std::free(part_nnz);
        cmtx::clear(&workspace);
        throw std::runtime_error("load_part_window_coo failed");
    }
    out.compressed_parts = convert_window_parts_to_csr(coo_view, &workspace);
    out.blocked_parts = convert_window_parts_to_blocked_ell(out.compressed_parts, cfg.block_size);
    out.rows = entry.slice_rows;
    out.cols = header.cols;
    out.nnz = 0u;
    for (const compressed_owned &compressed : out.compressed_parts) {
        out.nnz += compressed.part != nullptr ? compressed.part->nnz : 0u;
    }
    std::free(part_nnz);
    cs::clear(&coo_view);
    cmtx::clear(&workspace);
    return out;
}

static std::uint32_t count_valid_block_slots_local(const cs::sparse::blocked_ell *part, std::uint32_t row_block) {
    const std::uint32_t width = cs::sparse::ell_width_blocks(part);
    std::uint32_t count = 0u;
    if (part == nullptr || part->blockColIdx == nullptr || row_block >= cs::sparse::row_block_count(part)) return 0u;
    for (std::uint32_t slot = 0u; slot < width; ++slot) {
        if (part->blockColIdx[static_cast<std::size_t>(row_block) * width + slot] != cs::sparse::blocked_ell_invalid_col) {
            ++count;
        }
    }
    return count;
}

static std::uint32_t rows_in_row_block_local(const cs::sparse::blocked_ell *part, std::uint32_t row_block) {
    const std::uint32_t row_begin = part != nullptr ? row_block * part->block_size : 0u;
    if (part == nullptr || row_begin >= part->rows) return 0u;
    return std::min<std::uint32_t>(part->block_size, part->rows - row_begin);
}

static inline std::uint32_t popcount_u64(std::uint64_t x) {
    return static_cast<std::uint32_t>(__builtin_popcountll(static_cast<unsigned long long>(x)));
}

static inline const std::uint64_t *column_bits_ptr(const support_model &model, std::uint32_t col) {
    return model.column_bits.data() + static_cast<std::size_t>(col) * model.word_count;
}

static support_model build_support_model(const std::vector<blocked_ell_owned> &blocked_parts,
                                         std::uint32_t cols,
                                         std::uint32_t block_size) {
    support_model model;
    std::uint32_t row_block_cursor = 0u;
    model.cols = cols;
    model.block_size = block_size;
    model.parts.resize(blocked_parts.size());
    for (std::size_t i = 0; i < blocked_parts.size(); ++i) {
        const cs::sparse::blocked_ell *part = blocked_parts[i].part;
        const std::uint32_t row_block_count = part != nullptr ? cs::sparse::row_block_count(part) : 0u;
        model.parts[i].part = part;
        model.parts[i].row_block_begin = row_block_cursor;
        model.parts[i].row_block_count = row_block_count;
        row_block_cursor += row_block_count;
    }
    model.total_row_blocks = row_block_cursor;
    model.word_count = (row_block_cursor + 63u) / 64u;
    model.column_bits.assign(static_cast<std::size_t>(cols) * model.word_count, 0u);
    model.support_count.assign(cols, 0u);
    model.first_row_block.assign(cols, std::numeric_limits<std::uint32_t>::max());

    for (std::size_t part_i = 0; part_i < blocked_parts.size(); ++part_i) {
        const cs::sparse::blocked_ell *part = blocked_parts[part_i].part;
        const std::uint32_t width_blocks = part != nullptr ? cs::sparse::ell_width_blocks(part) : 0u;
        const std::uint32_t row_block_begin = model.parts[part_i].row_block_begin;
        if (part == nullptr) continue;
        for (std::uint32_t row_block = 0u; row_block < model.parts[part_i].row_block_count; ++row_block) {
            const std::uint32_t rows_in_block =
                std::min<std::uint32_t>(part->block_size, part->rows - row_block * part->block_size);
            const std::uint32_t global_row_block = row_block_begin + row_block;
            const std::uint32_t word = global_row_block / 64u;
            const std::uint64_t mask = 1ull << (global_row_block % 64u);
            for (std::uint32_t slot = 0u; slot < width_blocks; ++slot) {
                const std::uint32_t block_col = part->blockColIdx[static_cast<std::size_t>(row_block) * width_blocks + slot];
                if (block_col == cs::sparse::blocked_ell_invalid_col) continue;
                for (std::uint32_t col_in_block = 0u; col_in_block < part->block_size; ++col_in_block) {
                    const std::uint32_t col = block_col * part->block_size + col_in_block;
                    bool seen = false;
                    if (col >= cols) continue;
                    for (std::uint32_t row_in_block = 0u; row_in_block < rows_in_block; ++row_in_block) {
                        const std::size_t offset =
                            static_cast<std::size_t>(row_block * part->block_size + row_in_block) * part->ell_cols
                            + static_cast<std::size_t>(slot) * part->block_size + col_in_block;
                        if (__half2float(part->val[offset]) != 0.0f) {
                            seen = true;
                            break;
                        }
                    }
                    if (!seen) continue;
                    std::uint64_t *bits = model.column_bits.data() + static_cast<std::size_t>(col) * model.word_count;
                    if ((bits[word] & mask) == 0u) {
                        bits[word] |= mask;
                        model.support_count[col] += 1u;
                        model.first_row_block[col] = std::min(model.first_row_block[col], global_row_block);
                    }
                }
            }
        }
    }

    for (std::uint32_t col = 0u; col < cols; ++col) {
        if (model.first_row_block[col] == std::numeric_limits<std::uint32_t>::max()) model.first_row_block[col] = model.total_row_blocks;
    }
    return model;
}

static std::uint32_t bitset_intersection_count(const std::uint64_t *lhs,
                                               const std::uint64_t *rhs,
                                               std::uint32_t word_count) {
    std::uint32_t count = 0u;
    for (std::uint32_t w = 0u; w < word_count; ++w) count += popcount_u64(lhs[w] & rhs[w]);
    return count;
}

static std::vector<std::uint32_t> build_overlap_cluster_order(const support_model &model) {
    std::vector<std::uint32_t> seed_order(model.cols);
    std::vector<unsigned char> assigned(model.cols, 0u);
    std::vector<std::uint32_t> exec_to_canonical;
    exec_to_canonical.reserve(model.cols);
    std::iota(seed_order.begin(), seed_order.end(), 0u);
    std::stable_sort(seed_order.begin(),
                     seed_order.end(),
                     [&](std::uint32_t lhs, std::uint32_t rhs) {
                         if (model.support_count[lhs] != model.support_count[rhs]) {
                             return model.support_count[lhs] > model.support_count[rhs];
                         }
                         if (model.first_row_block[lhs] != model.first_row_block[rhs]) {
                             return model.first_row_block[lhs] < model.first_row_block[rhs];
                         }
                         return lhs < rhs;
                     });

    for (std::uint32_t seed : seed_order) {
        std::vector<std::pair<std::int64_t, std::uint32_t>> best;
        const std::uint64_t *seed_bits = column_bits_ptr(model, seed);
        if (assigned[seed] != 0u) continue;
        assigned[seed] = 1u;
        exec_to_canonical.push_back(seed);
        if (model.block_size <= 1u) continue;
        for (std::uint32_t col = 0u; col < model.cols; ++col) {
            if (assigned[col] != 0u) continue;
            const std::uint32_t overlap = bitset_intersection_count(seed_bits, column_bits_ptr(model, col), model.word_count);
            const std::uint32_t support = model.support_count[col];
            const std::int64_t score =
                static_cast<std::int64_t>(overlap) * 1048576ll
                - static_cast<std::int64_t>(support - overlap) * 4096ll
                - static_cast<std::int64_t>(model.first_row_block[col]);
            best.emplace_back(score, col);
        }
        std::stable_sort(best.begin(),
                         best.end(),
                         [](const auto &lhs, const auto &rhs) {
                             if (lhs.first != rhs.first) return lhs.first > rhs.first;
                             return lhs.second < rhs.second;
                         });
        const std::uint32_t take = std::min<std::uint32_t>(model.block_size - 1u, static_cast<std::uint32_t>(best.size()));
        for (std::uint32_t i = 0u; i < take; ++i) {
            assigned[best[i].second] = 1u;
            exec_to_canonical.push_back(best[i].second);
        }
    }

    for (std::uint32_t col = 0u; col < model.cols; ++col) {
        if (assigned[col] == 0u) exec_to_canonical.push_back(col);
    }
    return exec_to_canonical;
}

static std::vector<std::uint64_t> build_group_bits(const support_model &model,
                                                   const std::vector<std::uint32_t> &exec_to_canonical) {
    const std::uint32_t group_count = (model.cols + model.block_size - 1u) / model.block_size;
    std::vector<std::uint64_t> group_bits(static_cast<std::size_t>(group_count) * model.word_count, 0u);
    for (std::uint32_t exec_col = 0u; exec_col < model.cols; ++exec_col) {
        const std::uint32_t canonical_col = exec_to_canonical[exec_col];
        const std::uint32_t group = exec_col / model.block_size;
        const std::uint64_t *col_bits = column_bits_ptr(model, canonical_col);
        std::uint64_t *dst = group_bits.data() + static_cast<std::size_t>(group) * model.word_count;
        for (std::uint32_t w = 0u; w < model.word_count; ++w) dst[w] |= col_bits[w];
    }
    return group_bits;
}

static std::uint64_t segment_storage_bytes(std::uint32_t row_block_count,
                                           std::uint32_t rows,
                                           std::uint32_t width_blocks,
                                           std::uint32_t block_size) {
    return static_cast<std::uint64_t>(1u + 5u * sizeof(std::uint32_t))
        + static_cast<std::uint64_t>(row_block_count) * width_blocks * sizeof(std::uint32_t)
        + static_cast<std::uint64_t>(rows) * width_blocks * block_size * sizeof(::real::storage_t);
}

static std::uint64_t packed_u32_bytes(const std::uint32_t *values, std::size_t count, int allow_identity) {
    std::uint32_t max_value = 0u;
    bool identity = allow_identity != 0;
    for (std::size_t i = 0u; i < count; ++i) {
        const std::uint32_t value = values[i];
        if (identity && value != static_cast<std::uint32_t>(i)) identity = false;
        if (value > max_value) max_value = value;
    }
    if (identity) return sizeof(std::uint32_t);
    if (max_value <= 0xffu) return sizeof(std::uint32_t) + count * sizeof(std::uint8_t);
    if (max_value <= 0xffffu) return sizeof(std::uint32_t) + count * sizeof(std::uint16_t);
    return sizeof(std::uint32_t) + count * sizeof(std::uint32_t);
}

static std::uint64_t exact_segment_cost(std::uint32_t width_blocks,
                                        std::uint32_t row_blocks,
                                        std::uint32_t rows,
                                        std::uint32_t block_size) {
    return segment_storage_bytes(row_blocks, rows, width_blocks, block_size);
}

static dp_layout build_exact_dp_layout(const cs::sparse::blocked_ell *part, std::uint32_t bucket_cap) {
    dp_layout layout;
    const std::uint32_t row_blocks = part != nullptr ? cs::sparse::row_block_count(part) : 0u;
    const std::uint32_t full_rows = part != nullptr ? part->block_size : 0u;
    const bool has_partial =
        row_blocks != 0u && part != nullptr
        && std::min<std::uint32_t>(part->block_size, part->rows - (row_blocks - 1u) * part->block_size) != full_rows;
    const std::uint32_t sortable = has_partial ? row_blocks - 1u : row_blocks;
    const std::uint32_t partial_rows = has_partial
        ? std::min<std::uint32_t>(part->block_size, part->rows - (row_blocks - 1u) * part->block_size)
        : 0u;
    const std::uint32_t partial_width = has_partial ? count_valid_block_slots_local(part, row_blocks - 1u) : 0u;
    std::vector<std::uint32_t> widths(sortable, 0u);
    std::vector<std::uint32_t> order(sortable, 0u);
    std::vector<std::uint64_t> prefix_rows(static_cast<std::size_t>(sortable) + 1u, 0u);
    if (part == nullptr) return layout;

    for (std::uint32_t rb = 0u; rb < sortable; ++rb) {
        order[rb] = rb;
        widths[rb] = count_valid_block_slots_local(part, rb);
    }
    std::stable_sort(order.begin(),
                     order.end(),
                     [&](std::uint32_t lhs, std::uint32_t rhs) {
                         if (widths[lhs] != widths[rhs]) return widths[lhs] < widths[rhs];
                         return lhs < rhs;
                     });

    std::vector<std::uint32_t> sorted_widths(sortable, 0u);
    for (std::uint32_t i = 0u; i < sortable; ++i) {
        sorted_widths[i] = widths[order[i]];
        prefix_rows[i + 1u] = prefix_rows[i] + full_rows;
    }

    const std::uint32_t max_buckets = std::max<std::uint32_t>(1u, std::min<std::uint32_t>(bucket_cap, row_blocks == 0u ? 1u : row_blocks));
    const std::uint64_t inf = std::numeric_limits<std::uint64_t>::max() / 4u;
    std::vector<std::uint64_t> dp(static_cast<std::size_t>(max_buckets + 1u) * (sortable + 1u), inf);
    std::vector<std::int32_t> parent(static_cast<std::size_t>(max_buckets + 1u) * (sortable + 1u), -1);
    auto idx = [&](std::uint32_t b, std::uint32_t j) -> std::size_t {
        return static_cast<std::size_t>(b) * (sortable + 1u) + j;
    };

    dp[idx(0u, 0u)] = 0u;
    if (sortable == 0u) {
        layout.row_block_order.clear();
        if (has_partial) layout.row_block_order.push_back(row_blocks - 1u);
        layout.segment_row_block_offsets = {0u, has_partial ? 1u : 0u};
        layout.segment_width_blocks = {has_partial ? partial_width : 0u};
        layout.total_bytes = has_partial
            ? exact_segment_cost(partial_width, 1u, partial_rows, part->block_size)
            : 0u;
        return layout;
    }

    for (std::uint32_t buckets = 1u; buckets <= max_buckets; ++buckets) {
        for (std::uint32_t j = 1u; j <= sortable; ++j) {
            for (std::uint32_t i = buckets - 1u; i < j; ++i) {
                const bool include_partial = has_partial && j == sortable;
                const std::uint32_t width_blocks = include_partial
                    ? std::max<std::uint32_t>(sorted_widths[j - 1u], partial_width)
                    : sorted_widths[j - 1u];
                const std::uint32_t local_row_blocks = include_partial ? (j - i + 1u) : (j - i);
                const std::uint32_t local_rows = static_cast<std::uint32_t>(prefix_rows[j] - prefix_rows[i])
                    + (include_partial ? partial_rows : 0u);
                const std::uint64_t candidate = dp[idx(buckets - 1u, i)] + exact_segment_cost(width_blocks,
                                                                                               local_row_blocks,
                                                                                               local_rows,
                                                                                               part->block_size);
                if (candidate < dp[idx(buckets, j)]) {
                    dp[idx(buckets, j)] = candidate;
                    parent[idx(buckets, j)] = static_cast<std::int32_t>(i);
                }
            }
        }
    }

    std::uint32_t best_buckets = 1u;
    std::uint64_t best_bytes = dp[idx(1u, sortable)];
    for (std::uint32_t buckets = 1u; buckets <= max_buckets; ++buckets) {
        const std::uint64_t bytes = dp[idx(buckets, sortable)];
        if (bytes < best_bytes || (bytes == best_bytes && buckets < best_buckets)) {
            best_bytes = bytes;
            best_buckets = buckets;
        }
    }

    std::vector<std::uint32_t> boundaries;
    std::uint32_t j = sortable;
    std::uint32_t buckets = best_buckets;
    boundaries.push_back(has_partial ? row_blocks : sortable);
    while (buckets > 0u && j > 0u) {
        const std::int32_t i = parent[idx(buckets, j)];
        if (i < 0) break;
        boundaries.push_back(static_cast<std::uint32_t>(i));
        j = static_cast<std::uint32_t>(i);
        --buckets;
    }
    boundaries.push_back(0u);
    std::sort(boundaries.begin(), boundaries.end());
    boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());

    layout.row_block_order = order;
    if (has_partial) layout.row_block_order.push_back(row_blocks - 1u);
    layout.segment_row_block_offsets = boundaries;
    layout.segment_width_blocks.resize(layout.segment_row_block_offsets.size() - 1u, 0u);
    for (std::size_t seg = 0u; seg + 1u < layout.segment_row_block_offsets.size(); ++seg) {
        const std::uint32_t begin = layout.segment_row_block_offsets[seg];
        const std::uint32_t end = layout.segment_row_block_offsets[seg + 1u];
        std::uint32_t width = 0u;
        for (std::uint32_t pos = begin; pos < end; ++pos) {
            const std::uint32_t rb = layout.row_block_order[pos];
            width = std::max(width, count_valid_block_slots_local(part, rb));
        }
        layout.segment_width_blocks[seg] = width;
    }
    layout.total_bytes = best_bytes;
    return layout;
}

static int build_bucketed_partition_from_layout(const cs::sparse::blocked_ell *src,
                                                const dp_layout &layout,
                                                cs::bucketed_blocked_ell_partition *out,
                                                std::uint64_t *payload_bytes_out) {
    if (src == nullptr || out == nullptr) return 0;
    cs::clear(out);
    cs::init(out);
    out->rows = src->rows;
    out->cols = src->cols;
    out->nnz = src->nnz;
    out->segment_count = layout.segment_width_blocks.empty() ? 0u : static_cast<std::uint32_t>(layout.segment_width_blocks.size());
    out->segments = out->segment_count != 0u
        ? static_cast<cs::sparse::blocked_ell *>(std::calloc(out->segment_count, sizeof(cs::sparse::blocked_ell)))
        : nullptr;
    out->segment_row_offsets = static_cast<std::uint32_t *>(std::calloc(static_cast<std::size_t>(out->segment_count) + 1u,
                                                                        sizeof(std::uint32_t)));
    out->exec_to_canonical_rows = out->rows != 0u
        ? static_cast<std::uint32_t *>(std::calloc(out->rows, sizeof(std::uint32_t)))
        : nullptr;
    out->canonical_to_exec_rows = out->rows != 0u
        ? static_cast<std::uint32_t *>(std::calloc(out->rows, sizeof(std::uint32_t)))
        : nullptr;
    if ((out->segment_count != 0u && (out->segments == nullptr || out->segment_row_offsets == nullptr))
        || (out->rows != 0u && (out->exec_to_canonical_rows == nullptr || out->canonical_to_exec_rows == nullptr))) {
        cs::clear(out);
        return 0;
    }

    const std::uint32_t src_width_blocks = cs::sparse::ell_width_blocks(src);
    std::uint32_t row_cursor = 0u;
    std::uint64_t payload_bytes = 0u;
    for (std::uint32_t seg = 0u; seg < out->segment_count; ++seg) {
        const std::uint32_t begin = layout.segment_row_block_offsets[seg];
        const std::uint32_t end = layout.segment_row_block_offsets[seg + 1u];
        const std::uint32_t seg_width = layout.segment_width_blocks[seg];
        std::uint32_t seg_rows = 0u;
        for (std::uint32_t pos = begin; pos < end; ++pos) {
            const std::uint32_t rb = layout.row_block_order[pos];
            seg_rows += rows_in_row_block_local(src, rb);
        }
        out->segment_row_offsets[seg] = row_cursor;
        row_cursor += seg_rows;
        cs::sparse::init(out->segments + seg,
                         seg_rows,
                         src->cols,
                         seg_rows * seg_width * src->block_size,
                         src->block_size,
                         seg_width * src->block_size);
        if (!cs::sparse::allocate(out->segments + seg)) {
            cs::clear(out);
            return 0;
        }
        std::memset(out->segments[seg].storage, 0, cs::sparse::bytes(out->segments + seg) - sizeof(cs::sparse::blocked_ell));
        for (std::size_t idx = 0u;
             idx < static_cast<std::size_t>(cs::sparse::row_block_count(out->segments + seg)) * seg_width;
             ++idx) {
            out->segments[seg].blockColIdx[idx] = cs::sparse::blocked_ell_invalid_col;
        }
        payload_bytes += segment_storage_bytes(cs::sparse::row_block_count(out->segments + seg),
                                               out->segments[seg].rows,
                                               seg_width,
                                               src->block_size);
    }
    out->segment_row_offsets[out->segment_count] = row_cursor;

    for (std::uint32_t seg = 0u; seg < out->segment_count; ++seg) {
        const std::uint32_t begin = layout.segment_row_block_offsets[seg];
        const std::uint32_t end = layout.segment_row_block_offsets[seg + 1u];
        std::uint32_t exec_row = out->segment_row_offsets[seg];
        for (std::uint32_t pos = begin; pos < end; ++pos) {
            const std::uint32_t rb = layout.row_block_order[pos];
            const std::uint32_t rows_in_block = rows_in_row_block_local(src, rb);
            const std::uint32_t dst_row_block = pos - begin;
            std::uint32_t dst_slot = 0u;
            for (std::uint32_t row_in_block = 0u; row_in_block < rows_in_block; ++row_in_block) {
                const std::uint32_t canonical_row = rb * src->block_size + row_in_block;
                out->exec_to_canonical_rows[exec_row + row_in_block] = canonical_row;
                out->canonical_to_exec_rows[canonical_row] = exec_row + row_in_block;
            }
            for (std::uint32_t src_slot = 0u; src_slot < src_width_blocks; ++src_slot) {
                const std::uint32_t block_col =
                    src->blockColIdx[static_cast<std::size_t>(rb) * src_width_blocks + src_slot];
                if (block_col == cs::sparse::blocked_ell_invalid_col) continue;
                out->segments[seg].blockColIdx[static_cast<std::size_t>(dst_row_block) * layout.segment_width_blocks[seg] + dst_slot] =
                    block_col;
                for (std::uint32_t row_in_block = 0u; row_in_block < rows_in_block; ++row_in_block) {
                    const std::size_t src_offset =
                        static_cast<std::size_t>(rb * src->block_size + row_in_block) * src->ell_cols
                        + static_cast<std::size_t>(src_slot) * src->block_size;
                    const std::size_t dst_offset =
                        static_cast<std::size_t>(exec_row + row_in_block - out->segment_row_offsets[seg]) * out->segments[seg].ell_cols
                        + static_cast<std::size_t>(dst_slot) * src->block_size;
                    std::memcpy(out->segments[seg].val + dst_offset,
                                src->val + src_offset,
                                static_cast<std::size_t>(src->block_size) * sizeof(::real::storage_t));
                }
                ++dst_slot;
            }
            exec_row += rows_in_block;
        }
    }
    if (payload_bytes_out != nullptr) *payload_bytes_out = payload_bytes;
    return row_cursor == out->rows;
}

static std::uint64_t canonical_blocked_bytes(const std::vector<blocked_ell_owned> &parts) {
    std::uint64_t bytes = 0u;
    for (const blocked_ell_owned &part : parts) {
        if (part.part == nullptr) continue;
        bytes += static_cast<std::uint64_t>(cs::packed_bytes((const cs::sparse::blocked_ell *) nullptr,
                                                             part.part->rows,
                                                             part.part->cols,
                                                             part.part->nnz,
                                                             cs::partition_aux(part.part),
                                                             sizeof(::real::storage_t)));
    }
    return bytes;
}

static std::uint64_t csr_bytes(const std::vector<compressed_owned> &parts) {
    std::uint64_t bytes = 0u;
    for (const compressed_owned &part : parts) {
        if (part.part == nullptr) continue;
        bytes += cdataset::standard_csr_bytes(part.part->rows, part.part->nnz);
    }
    return bytes;
}

static width_eval evaluate_order_exact_dp(const support_model &model,
                                          const std::vector<std::uint32_t> &exec_to_canonical,
                                          std::uint32_t bucket_cap) {
    width_eval out;
    const std::vector<std::uint64_t> group_bits = build_group_bits(model, exec_to_canonical);
    const std::uint32_t group_count = (model.cols + model.block_size - 1u) / model.block_size;

    out.chosen_bucket_counts.resize(model.parts.size(), 0u);
    for (std::size_t part_i = 0u; part_i < model.parts.size(); ++part_i) {
        const part_descriptor &part_desc = model.parts[part_i];
        const std::uint32_t rows = part_desc.part != nullptr ? part_desc.part->rows : 0u;
        const std::uint32_t row_blocks = part_desc.row_block_count;
        const std::uint32_t block_size = part_desc.part != nullptr ? part_desc.part->block_size : model.block_size;
        const bool has_partial = row_blocks != 0u && rows % block_size != 0u;
        const std::uint32_t sortable = has_partial ? row_blocks - 1u : row_blocks;
        std::vector<std::uint32_t> widths(row_blocks, 0u);
        std::vector<std::uint32_t> sortable_widths(sortable, 0u);
        std::vector<std::uint32_t> sorted_widths(sortable, 0u);
        std::vector<std::uint32_t> order(sortable, 0u);
        std::vector<std::uint64_t> prefix_rows(static_cast<std::size_t>(sortable) + 1u, 0u);
        const std::uint32_t partial_rows = has_partial ? rows - sortable * block_size : 0u;

        for (std::uint32_t group = 0u; group < group_count; ++group) {
            const std::uint64_t *bits = group_bits.data() + static_cast<std::size_t>(group) * model.word_count;
            for (std::uint32_t word = 0u; word < model.word_count; ++word) {
                std::uint64_t live = bits[word];
                while (live != 0u) {
                    const std::uint32_t bit = static_cast<std::uint32_t>(__builtin_ctzll(static_cast<unsigned long long>(live)));
                    const std::uint32_t global_rb = word * 64u + bit;
                    if (global_rb >= part_desc.row_block_begin
                        && global_rb < part_desc.row_block_begin + part_desc.row_block_count) {
                        widths[global_rb - part_desc.row_block_begin] += 1u;
                    }
                    live &= (live - 1u);
                }
            }
        }
        for (std::uint32_t rb = 0u; rb < sortable; ++rb) {
            sortable_widths[rb] = widths[rb];
            order[rb] = rb;
            prefix_rows[rb + 1u] = prefix_rows[rb] + block_size;
        }
        std::stable_sort(order.begin(),
                         order.end(),
                         [&](std::uint32_t lhs, std::uint32_t rhs) {
                             if (sortable_widths[lhs] != sortable_widths[rhs]) return sortable_widths[lhs] < sortable_widths[rhs];
                             return lhs < rhs;
                         });
        for (std::uint32_t i = 0u; i < sortable; ++i) sorted_widths[i] = sortable_widths[order[i]];

        const std::uint32_t max_buckets = std::max<std::uint32_t>(1u, std::min<std::uint32_t>(bucket_cap, row_blocks == 0u ? 1u : row_blocks));
        const std::uint64_t inf = std::numeric_limits<std::uint64_t>::max() / 4u;
        std::vector<std::uint64_t> dp(static_cast<std::size_t>(max_buckets + 1u) * (sortable + 1u), inf);
        auto dp_index = [&](std::uint32_t b, std::uint32_t j) -> std::size_t {
            return static_cast<std::size_t>(b) * (sortable + 1u) + j;
        };
        dp[dp_index(0u, 0u)] = 0u;
        if (sortable == 0u) {
            const std::uint32_t width = has_partial ? widths[row_blocks - 1u] : 0u;
            out.objective_bytes += has_partial ? exact_segment_cost(width, 1u, partial_rows, block_size) : 0u;
            out.execution_payload_bytes += has_partial
                ? static_cast<std::uint64_t>(1u) * width * sizeof(std::uint32_t)
                    + static_cast<std::uint64_t>(partial_rows) * width * block_size * sizeof(::real::storage_t)
                : 0u;
            out.segment_index_bytes += has_partial ? static_cast<std::uint64_t>(width) * sizeof(std::uint32_t) : 0u;
            out.segment_value_bytes += has_partial
                ? static_cast<std::uint64_t>(partial_rows) * width * block_size * sizeof(::real::storage_t)
                : 0u;
            out.total_segments += has_partial ? 1u : 0u;
            out.chosen_bucket_counts[part_i] = has_partial ? 1u : 0u;
            continue;
        }
        for (std::uint32_t buckets = 1u; buckets <= max_buckets; ++buckets) {
            for (std::uint32_t j = 1u; j <= sortable; ++j) {
                for (std::uint32_t i = buckets - 1u; i < j; ++i) {
                    const bool include_partial = has_partial && j == sortable;
                    const std::uint32_t width = include_partial
                        ? std::max<std::uint32_t>(sorted_widths[j - 1u], widths[row_blocks - 1u])
                        : sorted_widths[j - 1u];
                    const std::uint32_t local_row_blocks = include_partial ? (j - i + 1u) : (j - i);
                    const std::uint32_t local_rows = static_cast<std::uint32_t>(prefix_rows[j] - prefix_rows[i])
                        + (include_partial ? partial_rows : 0u);
                    const std::uint64_t candidate =
                        dp[dp_index(buckets - 1u, i)] + exact_segment_cost(width, local_row_blocks, local_rows, block_size);
                    if (candidate < dp[dp_index(buckets, j)]) {
                        dp[dp_index(buckets, j)] = candidate;
                    }
                }
            }
        }
        std::uint32_t best_buckets = 1u;
        std::uint64_t best_bytes = dp[dp_index(1u, sortable)];
        for (std::uint32_t buckets = 1u; buckets <= max_buckets; ++buckets) {
            const std::uint64_t bytes = dp[dp_index(buckets, sortable)];
            if (bytes < best_bytes || (bytes == best_bytes && buckets < best_buckets)) {
                best_bytes = bytes;
                best_buckets = buckets;
            }
        }
        out.objective_bytes += best_bytes;
        out.chosen_bucket_counts[part_i] = best_buckets;
        out.total_segments += best_buckets;
        out.max_bucket_count = std::max(out.max_bucket_count, best_buckets);

        const std::uint32_t max_width = *std::max_element(widths.begin(), widths.end());
        std::uint64_t padded_slots = 0u;
        std::uint64_t used_slots = 0u;
        for (std::uint32_t rb = 0u; rb < row_blocks; ++rb) {
            const std::uint32_t rows_in_block = rb + 1u == row_blocks && has_partial ? partial_rows : block_size;
            used_slots += static_cast<std::uint64_t>(rows_in_block) * widths[rb] * block_size;
            padded_slots += static_cast<std::uint64_t>(rows_in_block) * max_width * block_size;
        }
        out.weighted_fill_ratio += padded_slots != 0u ? static_cast<double>(used_slots) / static_cast<double>(padded_slots) : 1.0;
    }
    if (!model.parts.empty()) out.weighted_fill_ratio /= static_cast<double>(model.parts.size());
    return out;
}

static std::vector<std::uint32_t> run_local_search(const support_model &model,
                                                   const std::vector<std::uint32_t> &seed_order,
                                                   std::uint32_t bucket_cap,
                                                   std::uint32_t max_passes) {
    std::vector<std::uint32_t> current = seed_order;
    width_eval best_eval = evaluate_order_exact_dp(model, current, bucket_cap);
    const std::uint32_t group_count = (model.cols + model.block_size - 1u) / model.block_size;

    for (std::uint32_t pass = 0u; pass < max_passes; ++pass) {
        bool any_improved = false;
        for (std::uint32_t group = 0u; group + 1u < group_count; ++group) {
            const std::uint32_t begin_a = group * model.block_size;
            const std::uint32_t end_a = std::min<std::uint32_t>(begin_a + model.block_size, model.cols);
            const std::uint32_t begin_b = end_a;
            const std::uint32_t end_b = std::min<std::uint32_t>(begin_b + model.block_size, model.cols);
            std::vector<std::uint64_t> bits_a(model.word_count, 0u);
            std::vector<std::uint64_t> bits_b(model.word_count, 0u);
            std::int64_t best_gain_a = std::numeric_limits<std::int64_t>::min();
            std::int64_t best_gain_b = std::numeric_limits<std::int64_t>::min();
            std::uint32_t swap_a = begin_a;
            std::uint32_t swap_b = begin_b;

            if (begin_b >= end_b) continue;
            for (std::uint32_t exec = begin_a; exec < end_a; ++exec) {
                const std::uint64_t *bits = column_bits_ptr(model, current[exec]);
                for (std::uint32_t w = 0u; w < model.word_count; ++w) bits_a[w] |= bits[w];
            }
            for (std::uint32_t exec = begin_b; exec < end_b; ++exec) {
                const std::uint64_t *bits = column_bits_ptr(model, current[exec]);
                for (std::uint32_t w = 0u; w < model.word_count; ++w) bits_b[w] |= bits[w];
            }
            for (std::uint32_t exec = begin_a; exec < end_a; ++exec) {
                const std::uint64_t *bits = column_bits_ptr(model, current[exec]);
                const std::int64_t own = bitset_intersection_count(bits, bits_a.data(), model.word_count);
                const std::int64_t neighbor = bitset_intersection_count(bits, bits_b.data(), model.word_count);
                const std::int64_t gain = neighbor - own;
                if (gain > best_gain_a) {
                    best_gain_a = gain;
                    swap_a = exec;
                }
            }
            for (std::uint32_t exec = begin_b; exec < end_b; ++exec) {
                const std::uint64_t *bits = column_bits_ptr(model, current[exec]);
                const std::int64_t own = bitset_intersection_count(bits, bits_b.data(), model.word_count);
                const std::int64_t neighbor = bitset_intersection_count(bits, bits_a.data(), model.word_count);
                const std::int64_t gain = neighbor - own;
                if (gain > best_gain_b) {
                    best_gain_b = gain;
                    swap_b = exec;
                }
            }
            if (swap_a >= current.size() || swap_b >= current.size()) continue;
            std::swap(current[swap_a], current[swap_b]);
            const width_eval candidate = evaluate_order_exact_dp(model, current, bucket_cap);
            if (candidate.objective_bytes < best_eval.objective_bytes) {
                best_eval = candidate;
                any_improved = true;
            } else {
                std::swap(current[swap_a], current[swap_b]);
            }
        }
        if (!any_improved) break;
    }
    return current;
}

static std::vector<std::uint32_t> invert_permutation(const std::vector<std::uint32_t> &exec_to_canonical) {
    std::vector<std::uint32_t> canonical_to_exec(exec_to_canonical.size(), 0u);
    for (std::uint32_t exec = 0u; exec < exec_to_canonical.size(); ++exec) {
        canonical_to_exec[exec_to_canonical[exec]] = exec;
    }
    return canonical_to_exec;
}

static int build_exact_dp_shard(const std::vector<blocked_ell_owned> &parts,
                                std::uint32_t cols,
                                const std::vector<std::uint32_t> &exec_to_canonical_cols,
                                std::uint32_t bucket_cap,
                                cs::bucketed_blocked_ell_shard *out,
                                width_eval *eval_out) {
    std::vector<std::uint32_t> canonical_to_exec_cols = invert_permutation(exec_to_canonical_cols);
    cs::sparse::coo canonical_coo;
    cs::sparse::blocked_ell permuted;
    std::uint32_t row_cursor = 0u;
    std::uint64_t execution_payload_bytes = 0u;
    std::uint64_t segment_value_bytes = 0u;
    std::uint64_t segment_index_bytes = 0u;
    std::uint32_t total_segments = 0u;
    std::uint32_t max_bucket_count = 0u;
    if (out == nullptr) return 0;
    cs::clear(out);
    cs::init(out);
    out->rows = 0u;
    out->cols = cols;
    out->nnz = 0u;
    out->partition_count = static_cast<std::uint32_t>(parts.size());
    out->partitions = parts.empty()
        ? nullptr
        : static_cast<cs::bucketed_blocked_ell_partition *>(std::calloc(parts.size(), sizeof(cs::bucketed_blocked_ell_partition)));
    out->partition_row_offsets = static_cast<std::uint32_t *>(std::calloc(parts.size() + 1u, sizeof(std::uint32_t)));
    out->exec_to_canonical_cols = cols != 0u ? static_cast<std::uint32_t *>(std::calloc(cols, sizeof(std::uint32_t))) : nullptr;
    out->canonical_to_exec_cols = cols != 0u ? static_cast<std::uint32_t *>(std::calloc(cols, sizeof(std::uint32_t))) : nullptr;
    if ((out->partition_count != 0u && (out->partitions == nullptr || out->partition_row_offsets == nullptr))
        || (cols != 0u && (out->exec_to_canonical_cols == nullptr || out->canonical_to_exec_cols == nullptr))) {
        cs::clear(out);
        return 0;
    }
    if (cols != 0u) {
        std::memcpy(out->exec_to_canonical_cols, exec_to_canonical_cols.data(), static_cast<std::size_t>(cols) * sizeof(std::uint32_t));
        std::memcpy(out->canonical_to_exec_cols, canonical_to_exec_cols.data(), static_cast<std::size_t>(cols) * sizeof(std::uint32_t));
    }

    cs::sparse::init(&canonical_coo);
    cs::sparse::init(&permuted);
    for (std::size_t part_i = 0u; part_i < parts.size(); ++part_i) {
        std::uint64_t payload_bytes = 0u;
        const cs::sparse::blocked_ell *src = parts[part_i].part;
        if (src == nullptr) continue;
        out->partition_row_offsets[part_i] = row_cursor;
        cs::init(out->partitions + part_i);
        if (!cdataset::blocked_ell_to_canonical_coo(src, &canonical_coo)
            || !cs::convert::blocked_ell_from_coo(&canonical_coo,
                                                  cols,
                                                  canonical_to_exec_cols.data(),
                                                  src->block_size,
                                                  &permuted)) {
            cs::sparse::clear(&canonical_coo);
            cs::sparse::clear(&permuted);
            cs::clear(out);
            return 0;
        }
        const dp_layout layout = build_exact_dp_layout(&permuted, bucket_cap);
        if (!build_bucketed_partition_from_layout(&permuted, layout, out->partitions + part_i, &payload_bytes)) {
            cs::sparse::clear(&canonical_coo);
            cs::sparse::clear(&permuted);
            cs::clear(out);
            return 0;
        }
        execution_payload_bytes += payload_bytes;
        for (std::uint32_t seg = 0u; seg < out->partitions[part_i].segment_count; ++seg) {
            const cs::sparse::blocked_ell *segment = out->partitions[part_i].segments + seg;
            const std::uint32_t width_blocks = cs::sparse::ell_width_blocks(segment);
            segment_index_bytes += static_cast<std::uint64_t>(cs::sparse::row_block_count(segment)) * width_blocks * sizeof(std::uint32_t);
            segment_value_bytes += static_cast<std::uint64_t>(segment->rows) * segment->ell_cols * sizeof(::real::storage_t);
        }
        out->partitions[part_i].exec_to_canonical_cols = cols != 0u
            ? static_cast<std::uint32_t *>(std::calloc(cols, sizeof(std::uint32_t)))
            : nullptr;
        out->partitions[part_i].canonical_to_exec_cols = cols != 0u
            ? static_cast<std::uint32_t *>(std::calloc(cols, sizeof(std::uint32_t)))
            : nullptr;
        if (cols != 0u && (out->partitions[part_i].exec_to_canonical_cols == nullptr
            || out->partitions[part_i].canonical_to_exec_cols == nullptr)) {
            cs::sparse::clear(&canonical_coo);
            cs::sparse::clear(&permuted);
            cs::clear(out);
            return 0;
        }
        if (cols != 0u) {
            std::memcpy(out->partitions[part_i].exec_to_canonical_cols, exec_to_canonical_cols.data(), static_cast<std::size_t>(cols) * sizeof(std::uint32_t));
            std::memcpy(out->partitions[part_i].canonical_to_exec_cols, canonical_to_exec_cols.data(), static_cast<std::size_t>(cols) * sizeof(std::uint32_t));
        }
        total_segments += out->partitions[part_i].segment_count;
        max_bucket_count = std::max(max_bucket_count, out->partitions[part_i].segment_count);
        row_cursor += out->partitions[part_i].rows;
        out->rows += out->partitions[part_i].rows;
        out->nnz += out->partitions[part_i].nnz;
        cs::sparse::clear(&canonical_coo);
        cs::sparse::init(&canonical_coo);
        cs::sparse::clear(&permuted);
        cs::sparse::init(&permuted);
    }
    out->partition_row_offsets[parts.size()] = row_cursor;
    if (eval_out != nullptr) {
        eval_out->execution_payload_bytes = execution_payload_bytes;
        eval_out->segment_value_bytes = segment_value_bytes;
        eval_out->segment_index_bytes = segment_index_bytes;
        eval_out->total_segments = total_segments;
        eval_out->max_bucket_count = max_bucket_count;
    }
    cs::sparse::clear(&canonical_coo);
    cs::sparse::clear(&permuted);
    return 1;
}

static std::uint64_t optimized_partition_blob_bytes(const cs::bucketed_blocked_ell_partition &part) {
    std::uint64_t bytes = 4u * sizeof(std::uint32_t);
    bytes += packed_u32_bytes(part.segment_row_offsets, static_cast<std::size_t>(part.segment_count) + 1u, 0);
    bytes += packed_u32_bytes(part.exec_to_canonical_rows, part.rows, 1);
    for (std::uint32_t seg = 0u; seg < part.segment_count; ++seg) {
        const cs::sparse::blocked_ell *segment = part.segments + seg;
        bytes += segment_storage_bytes(cs::sparse::row_block_count(segment),
                                       segment->rows,
                                       cs::sparse::ell_width_blocks(segment),
                                       segment->block_size);
    }
    return bytes;
}

static std::uint64_t optimized_shard_blob_bytes(const cs::bucketed_blocked_ell_shard &shard) {
    std::uint64_t bytes = 8u + 4u * sizeof(std::uint32_t);
    bytes += packed_u32_bytes(shard.partition_row_offsets, static_cast<std::size_t>(shard.partition_count) + 1u, 0);
    bytes += packed_u32_bytes(shard.exec_to_canonical_cols, shard.cols, 1);
    for (std::uint32_t part = 0u; part < shard.partition_count; ++part) {
        bytes += optimized_partition_blob_bytes(shard.partitions[part]);
    }
    return bytes;
}

static std::vector<partition_device_segments> upload_shard_segments(int device,
                                                                    const cs::bucketed_blocked_ell_shard &shard,
                                                                    std::uint32_t rhs_cols,
                                                                    autograd::device_buffer<float> *rhs_device) {
    std::vector<partition_device_segments> out(shard.partition_count);
    std::vector<float> host_rhs(static_cast<std::size_t>(shard.cols) * rhs_cols, 0.0f);
    for (std::size_t i = 0u; i < host_rhs.size(); ++i) host_rhs[i] = static_cast<float>((i % 17u) + 1u) / 17.0f;
    autograd::cuda_require(cudaSetDevice(device), "cudaSetDevice(upload_shard_segments)");
    *rhs_device = autograd::allocate_device_buffer<float>(host_rhs.size());
    autograd::upload_device_buffer(rhs_device, host_rhs.data(), host_rhs.size());

    for (std::uint32_t part = 0u; part < shard.partition_count; ++part) {
        out[part].segments.resize(shard.partitions[part].segment_count);
        for (std::uint32_t seg = 0u; seg < shard.partitions[part].segment_count; ++seg) {
            const cs::sparse::blocked_ell *src = shard.partitions[part].segments + seg;
            device_segment &dst = out[part].segments[seg];
            const std::size_t idx_count =
                static_cast<std::size_t>(cs::sparse::row_block_count(src)) * cs::sparse::ell_width_blocks(src);
            const std::size_t val_count = static_cast<std::size_t>(src->rows) * src->ell_cols;
            dst.rows = src->rows;
            dst.cols = src->cols;
            dst.block_size = src->block_size;
            dst.ell_cols = src->ell_cols;
            dst.block_col_idx = autograd::allocate_device_buffer<std::uint32_t>(idx_count);
            dst.values = autograd::allocate_device_buffer<__half>(val_count);
            dst.out = autograd::allocate_device_buffer<float>(static_cast<std::size_t>(src->rows) * rhs_cols);
            autograd::upload_device_buffer(&dst.block_col_idx, src->blockColIdx, idx_count);
            autograd::upload_device_buffer(&dst.values, src->val, val_count);
        }
    }
    return out;
}

static double benchmark_shard_spmm_ms(int device,
                                      const cs::bucketed_blocked_ell_shard &shard,
                                      std::uint32_t rhs_cols,
                                      const config &cfg) {
    autograd::execution_context ctx;
    autograd::device_buffer<float> rhs_device;
    auto uploaded = upload_shard_segments(device, shard, rhs_cols, &rhs_device);
    autograd::init(&ctx, device, nullptr);
    autograd::cuda_require(cudaSetDevice(device), "cudaSetDevice(benchmark_shard_spmm_ms)");

    auto run_once = [&]() {
        for (std::uint32_t part = 0u; part < shard.partition_count; ++part) {
            for (std::size_t seg = 0u; seg < uploaded[part].segments.size(); ++seg) {
                device_segment &matrix = uploaded[part].segments[seg];
                autograd::base::blocked_ell_spmm_fwd_f16_f32(ctx,
                                                             matrix.block_col_idx.data,
                                                             matrix.values.data,
                                                             matrix.rows,
                                                             matrix.cols,
                                                             matrix.block_size,
                                                             matrix.ell_cols,
                                                             rhs_device.data,
                                                             rhs_cols,
                                                             rhs_cols,
                                                             matrix.out.data,
                                                             rhs_cols);
            }
        }
    };

    for (std::uint32_t i = 0u; i < cfg.warmup; ++i) {
        run_once();
        autograd::cuda_require(cudaDeviceSynchronize(), "cudaDeviceSynchronize(spmm warmup)");
    }
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    autograd::cuda_require(cudaEventCreate(&start), "cudaEventCreate(start)");
    autograd::cuda_require(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    autograd::cuda_require(cudaEventRecord(start, nullptr), "cudaEventRecord(start)");
    for (std::uint32_t i = 0u; i < cfg.iters; ++i) run_once();
    autograd::cuda_require(cudaEventRecord(stop, nullptr), "cudaEventRecord(stop)");
    autograd::cuda_require(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float elapsed_ms = 0.0f;
    autograd::cuda_require(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    autograd::clear(&ctx);
    return static_cast<double>(elapsed_ms) / static_cast<double>(cfg.iters);
}

static algorithm_result run_baseline(const real_case &entry,
                                     const case_matrices &matrices,
                                     const support_model &model,
                                     const config &cfg) {
    algorithm_result result;
    cs::bucketed_blocked_ell_shard shard;
    auto build_begin = std::chrono::steady_clock::now();
    cs::init(&shard);
    std::vector<cs::sparse::blocked_ell> part_views;
    part_views.reserve(matrices.blocked_parts.size());
    for (const blocked_ell_owned &part : matrices.blocked_parts) part_views.push_back(*part.part);
    require(cdataset::build_bucketed_optimized_shard(part_views, matrices.cols, static_cast<int>(cfg.device), &shard),
            "build_bucketed_optimized_shard failed");
    auto build_end = std::chrono::steady_clock::now();
    result.algorithm = "baseline";
    result.dataset_id = entry.dataset_id;
    result.rows = matrices.rows;
    result.cols = matrices.cols;
    result.nnz = matrices.nnz;
    result.block_size = cfg.block_size;
    result.csr_bytes = csr_bytes(matrices.compressed_parts);
    result.canonical_blocked_bytes = canonical_blocked_bytes(matrices.blocked_parts);
    result.optimized_shard_bytes = optimized_shard_blob_bytes(shard);
    for (std::uint32_t part = 0u; part < shard.partition_count; ++part) {
        result.total_segments += shard.partitions[part].segment_count;
        result.max_bucket_count = std::max(result.max_bucket_count, shard.partitions[part].segment_count);
        for (std::uint32_t seg = 0u; seg < shard.partitions[part].segment_count; ++seg) {
            const cs::sparse::blocked_ell *segment = shard.partitions[part].segments + seg;
            const std::uint32_t width_blocks = cs::sparse::ell_width_blocks(segment);
            result.segment_index_bytes += static_cast<std::uint64_t>(cs::sparse::row_block_count(segment)) * width_blocks * sizeof(std::uint32_t);
            result.segment_value_bytes += static_cast<std::uint64_t>(segment->rows) * segment->ell_cols * sizeof(::real::storage_t);
        }
    }
    result.execution_payload_bytes = result.segment_index_bytes + result.segment_value_bytes;
    result.build_ms = std::chrono::duration<double, std::milli>(build_end - build_begin).count();
    result.canonical_vs_csr = result.csr_bytes != 0u
        ? static_cast<double>(result.canonical_blocked_bytes) / static_cast<double>(result.csr_bytes)
        : 0.0;
    result.optimized_vs_csr = result.csr_bytes != 0u
        ? static_cast<double>(result.optimized_shard_bytes) / static_cast<double>(result.csr_bytes)
        : 0.0;
    if (cfg.run_spmm != 0) result.spmm_ms = benchmark_shard_spmm_ms(static_cast<int>(cfg.device), shard, cfg.rhs_cols, cfg);
    cs::clear(&shard);
    (void) model;
    return result;
}

static algorithm_result run_exact_variant(const char *name,
                                          const real_case &entry,
                                          const case_matrices &matrices,
                                          const std::vector<std::uint32_t> &exec_to_canonical,
                                          const width_eval &eval,
                                          const config &cfg) {
    algorithm_result result;
    cs::bucketed_blocked_ell_shard shard;
    width_eval build_eval{};
    auto build_begin = std::chrono::steady_clock::now();
    cs::init(&shard);
    require(build_exact_dp_shard(matrices.blocked_parts,
                                 matrices.cols,
                                 exec_to_canonical,
                                 cfg.bucket_cap,
                                 &shard,
                                 &build_eval),
            "build_exact_dp_shard failed");
    auto build_end = std::chrono::steady_clock::now();
    result.algorithm = name;
    result.dataset_id = entry.dataset_id;
    result.rows = matrices.rows;
    result.cols = matrices.cols;
    result.nnz = matrices.nnz;
    result.block_size = cfg.block_size;
    result.csr_bytes = csr_bytes(matrices.compressed_parts);
    result.canonical_blocked_bytes = canonical_blocked_bytes(matrices.blocked_parts);
    result.optimized_shard_bytes = optimized_shard_blob_bytes(shard);
    result.execution_payload_bytes = build_eval.execution_payload_bytes;
    result.segment_value_bytes = build_eval.segment_value_bytes;
    result.segment_index_bytes = build_eval.segment_index_bytes;
    result.total_segments = build_eval.total_segments;
    result.max_bucket_count = build_eval.max_bucket_count;
    result.weighted_fill_ratio = eval.weighted_fill_ratio;
    result.build_ms = std::chrono::duration<double, std::milli>(build_end - build_begin).count();
    result.canonical_vs_csr = result.csr_bytes != 0u
        ? static_cast<double>(result.canonical_blocked_bytes) / static_cast<double>(result.csr_bytes)
        : 0.0;
    result.optimized_vs_csr = result.csr_bytes != 0u
        ? static_cast<double>(result.optimized_shard_bytes) / static_cast<double>(result.csr_bytes)
        : 0.0;
    if (cfg.run_spmm != 0) result.spmm_ms = benchmark_shard_spmm_ms(static_cast<int>(cfg.device), shard, cfg.rhs_cols, cfg);
    cs::clear(&shard);
    return result;
}

static void print_result(const algorithm_result &result) {
    std::cout
        << "algorithm=" << result.algorithm
        << " dataset=" << result.dataset_id
        << " rows=" << result.rows
        << " cols=" << result.cols
        << " nnz=" << result.nnz
        << " block_size=" << result.block_size
        << " csr_bytes=" << result.csr_bytes
        << " canonical_blocked_bytes=" << result.canonical_blocked_bytes
        << " optimized_shard_bytes=" << result.optimized_shard_bytes
        << " execution_payload_bytes=" << result.execution_payload_bytes
        << " total_segments=" << result.total_segments
        << " max_bucket_count=" << result.max_bucket_count
        << " canonical_vs_csr=" << std::fixed << std::setprecision(4) << result.canonical_vs_csr
        << " optimized_vs_csr=" << std::fixed << std::setprecision(4) << result.optimized_vs_csr
        << " weighted_fill=" << std::fixed << std::setprecision(4) << result.weighted_fill_ratio
        << " build_ms=" << std::fixed << std::setprecision(3) << result.build_ms
        << " spmm_ms=" << std::fixed << std::setprecision(3) << result.spmm_ms
        << "\n";
}

static std::string json_escape(const std::string &text) {
    std::string out;
    out.reserve(text.size() + 8u);
    for (char c : text) {
        if (c == '\\' || c == '"') {
            out.push_back('\\');
            out.push_back(c);
        } else if (c == '\n') {
            out += "\\n";
        } else {
            out.push_back(c);
        }
    }
    return out;
}

static void write_json_summary(const std::string &path, const std::vector<algorithm_result> &results) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("failed to open json output");
    out << "{\n  \"results\": [\n";
    for (std::size_t i = 0u; i < results.size(); ++i) {
        const algorithm_result &r = results[i];
        out << "    {\n"
            << "      \"algorithm\": \"" << json_escape(r.algorithm) << "\",\n"
            << "      \"dataset\": \"" << json_escape(r.dataset_id) << "\",\n"
            << "      \"rows\": " << r.rows << ",\n"
            << "      \"cols\": " << r.cols << ",\n"
            << "      \"nnz\": " << r.nnz << ",\n"
            << "      \"block_size\": " << r.block_size << ",\n"
            << "      \"csr_bytes\": " << r.csr_bytes << ",\n"
            << "      \"canonical_blocked_bytes\": " << r.canonical_blocked_bytes << ",\n"
            << "      \"optimized_shard_bytes\": " << r.optimized_shard_bytes << ",\n"
            << "      \"execution_payload_bytes\": " << r.execution_payload_bytes << ",\n"
            << "      \"segment_value_bytes\": " << r.segment_value_bytes << ",\n"
            << "      \"segment_index_bytes\": " << r.segment_index_bytes << ",\n"
            << "      \"total_segments\": " << r.total_segments << ",\n"
            << "      \"max_bucket_count\": " << r.max_bucket_count << ",\n"
            << "      \"weighted_fill_ratio\": " << std::fixed << std::setprecision(6) << r.weighted_fill_ratio << ",\n"
            << "      \"canonical_vs_csr\": " << std::fixed << std::setprecision(6) << r.canonical_vs_csr << ",\n"
            << "      \"optimized_vs_csr\": " << std::fixed << std::setprecision(6) << r.optimized_vs_csr << ",\n"
            << "      \"build_ms\": " << std::fixed << std::setprecision(6) << r.build_ms << ",\n"
            << "      \"spmm_ms\": " << std::fixed << std::setprecision(6) << r.spmm_ms << "\n"
            << "    }" << (i + 1u == results.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
}

} // namespace

int main(int argc, char **argv) {
    try {
        const config cfg = parse_args(argc, argv);
        const cellerator::bench::benchmark_mutex_guard guard("blockedEllStudyBench");
        const std::vector<real_case> cases = load_manifest(cfg.manifest_path, cfg.dataset_filter);
        std::vector<algorithm_result> all_results;

        for (const real_case &entry : cases) {
            const case_matrices matrices = load_case_matrices(entry, cfg);
            const support_model model = build_support_model(matrices.blocked_parts, matrices.cols, cfg.block_size);
            std::vector<cs::sparse::blocked_ell> part_views;
            std::vector<std::uint32_t> baseline_exec_to_canonical;
            std::vector<std::uint32_t> baseline_canonical_to_exec;

            part_views.reserve(matrices.blocked_parts.size());
            for (const blocked_ell_owned &part : matrices.blocked_parts) part_views.push_back(*part.part);
            require(cdataset::build_shard_column_maps(part_views,
                                                      matrices.cols,
                                                      &baseline_exec_to_canonical,
                                                      &baseline_canonical_to_exec),
                    "build_shard_column_maps failed");

            const std::vector<std::uint32_t> overlap_order = build_overlap_cluster_order(model);
            const std::vector<std::uint32_t> local_order =
                run_local_search(model, overlap_order, cfg.bucket_cap, cfg.local_search_passes);
            const width_eval exact_eval = evaluate_order_exact_dp(model, baseline_exec_to_canonical, cfg.bucket_cap);
            const width_eval overlap_eval = evaluate_order_exact_dp(model, overlap_order, cfg.bucket_cap);
            const width_eval local_eval = evaluate_order_exact_dp(model, local_order, cfg.bucket_cap);

            if (cfg.algorithm_filter == "all" || cfg.algorithm_filter == "baseline") {
                all_results.push_back(run_baseline(entry, matrices, model, cfg));
                print_result(all_results.back());
            }
            if (cfg.algorithm_filter == "all" || cfg.algorithm_filter == "exact-dp") {
                all_results.push_back(run_exact_variant("exact-dp",
                                                        entry,
                                                        matrices,
                                                        baseline_exec_to_canonical,
                                                        exact_eval,
                                                        cfg));
                print_result(all_results.back());
            }
            if (cfg.algorithm_filter == "all" || cfg.algorithm_filter == "overlap") {
                all_results.push_back(run_exact_variant("overlap",
                                                        entry,
                                                        matrices,
                                                        overlap_order,
                                                        overlap_eval,
                                                        cfg));
                print_result(all_results.back());
            }
            if (cfg.algorithm_filter == "all" || cfg.algorithm_filter == "local-search") {
                all_results.push_back(run_exact_variant("local-search",
                                                        entry,
                                                        matrices,
                                                        local_order,
                                                        local_eval,
                                                        cfg));
                print_result(all_results.back());
            }
        }

        if (!cfg.json_path.empty()) write_json_summary(cfg.json_path, all_results);
        return 0;
    } catch (const std::exception &err) {
        std::fprintf(stderr, "blockedEllStudyBench: %s\n", err.what());
        return 1;
    }
}
