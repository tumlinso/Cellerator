#include "benchmark_mutex.hh"
#include "../extern/CellShard/src/CellShard.hh"
#include "../src/ingest/mtx/compressed_parts.cuh"
#include "../src/ingest/mtx/mtx_reader.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace cs = ::cellshard;
namespace cmtx = ::cellerator::ingest::mtx;
namespace csc = ::cellshard::convert;

struct real_case {
    std::string dataset_id;
    std::string matrix_path;
    std::uint32_t rows = 0u;
    std::uint32_t cols = 0u;
    std::uint64_t nnz = 0u;
    std::uint32_t slice_rows = 0u;
};

struct config {
    std::string manifest_path = "bench/real_data/embryo_spmm_manifest.tsv";
    std::string dataset_filter = "all";
    unsigned int device = 0u;
    unsigned int parts = 4u;
    unsigned int reader_bytes_mb = 16u;
    unsigned int repeats = 3u;
    unsigned int seed = 7u;
    unsigned int ideal_block_size = 16u;
    std::vector<unsigned int> candidates = {8u, 16u, 32u};
    std::vector<double> levels = {1.0, 0.9, 0.75, 0.6, 0.45};
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

struct compressed_case {
    std::vector<compressed_owned> parts;
    std::uint64_t rows = 0u;
    std::uint32_t cols = 0u;
    std::uint64_t nnz = 0u;
};

struct build_metrics {
    std::string source;
    std::string dataset_id;
    std::string level_label;
    std::uint64_t rows = 0u;
    std::uint32_t cols = 0u;
    std::uint64_t nnz = 0u;
    std::uint32_t parts = 0u;
    std::uint32_t ideal_block_size = 0u;
    std::uint32_t tuned_block_size = 0u;
    double tuned_fill_ratio = 0.0;
    double weighted_fill_ratio = 0.0;
    std::size_t tuned_padded_bytes = 0u;
    std::size_t padded_bytes = 0u;
    double storage_amplification = 0.0;
    double build_ms_total = 0.0;
    double build_ms_per_part = 0.0;
};

static int check_cuda(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "CUDA error at %s: %s\n", label, cudaGetErrorString(err));
    return 0;
}

static void usage(const char *argv0) {
    std::fprintf(stderr,
                 "Usage: %s [options]\n"
                 "  --manifest PATH         Real-data manifest. Default: bench/real_data/embryo_spmm_manifest.tsv\n"
                 "  --dataset ID|all        Dataset filter. Default: all\n"
                 "  --device N              CUDA device for COO->CSR staging. Default: 0\n"
                 "  --parts N               Part count for mirrored real/synthetic slices. Default: 4\n"
                 "  --reader-bytes-mb N     MTX reader buffer in MiB. Default: 16\n"
                 "  --repeats N             Build repeats per case. Default: 3\n"
                 "  --seed N                RNG seed. Default: 7\n"
                 "  --ideal-block-size N    Synthetic template block size. Default: 16\n"
                 "  --candidates CSV        Block-size candidates. Default: 8,16,32\n"
                 "  --levels CSV            Synthetic perfection levels in [0,1]. Default: 1.0,0.9,0.75,0.6,0.45\n",
                 argv0);
}

static std::uint32_t parse_u32(const char *text, const char *label) {
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (text == nullptr || *text == '\0' || end == nullptr || *end != '\0' || parsed > 0xfffffffful) {
        throw std::invalid_argument(std::string("invalid integer for ") + label);
    }
    return (std::uint32_t) parsed;
}

static double parse_f64(const char *text, const char *label) {
    char *end = nullptr;
    const double parsed = std::strtod(text, &end);
    if (text == nullptr || *text == '\0' || end == nullptr || *end != '\0') {
        throw std::invalid_argument(std::string("invalid floating-point value for ") + label);
    }
    return parsed;
}

static std::vector<std::string> split_csv(const std::string &text) {
    std::vector<std::string> out;
    std::size_t begin = 0u;
    while (begin <= text.size()) {
        const std::size_t end = text.find(',', begin);
        if (end == std::string::npos) {
            out.push_back(text.substr(begin));
            break;
        }
        out.push_back(text.substr(begin, end - begin));
        begin = end + 1u;
    }
    return out;
}

static std::vector<unsigned int> parse_u32_csv(const char *text, const char *label) {
    std::vector<unsigned int> out;
    for (const std::string &token : split_csv(text != nullptr ? std::string(text) : std::string())) {
        if (token.empty()) continue;
        out.push_back(parse_u32(token.c_str(), label));
    }
    if (out.empty()) throw std::invalid_argument(std::string("empty CSV list for ") + label);
    return out;
}

static std::vector<double> parse_f64_csv(const char *text, const char *label) {
    std::vector<double> out;
    for (const std::string &token : split_csv(text != nullptr ? std::string(text) : std::string())) {
        if (token.empty()) continue;
        const double value = parse_f64(token.c_str(), label);
        if (!(value >= 0.0 && value <= 1.0)) {
            throw std::invalid_argument(std::string(label) + " values must be in [0,1]");
        }
        out.push_back(value);
    }
    if (out.empty()) throw std::invalid_argument(std::string("empty CSV list for ") + label);
    return out;
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
        } else if (arg == "--device") {
            cfg.device = parse_u32(require_value("--device"), "--device");
        } else if (arg == "--parts") {
            cfg.parts = parse_u32(require_value("--parts"), "--parts");
        } else if (arg == "--reader-bytes-mb") {
            cfg.reader_bytes_mb = parse_u32(require_value("--reader-bytes-mb"), "--reader-bytes-mb");
        } else if (arg == "--repeats") {
            cfg.repeats = parse_u32(require_value("--repeats"), "--repeats");
        } else if (arg == "--seed") {
            cfg.seed = parse_u32(require_value("--seed"), "--seed");
        } else if (arg == "--ideal-block-size") {
            cfg.ideal_block_size = parse_u32(require_value("--ideal-block-size"), "--ideal-block-size");
        } else if (arg == "--candidates") {
            cfg.candidates = parse_u32_csv(require_value("--candidates"), "--candidates");
        } else if (arg == "--levels") {
            cfg.levels = parse_f64_csv(require_value("--levels"), "--levels");
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument(std::string("unknown argument: ") + arg);
        }
    }

    if (cfg.parts == 0u) throw std::invalid_argument("--parts must be non-zero");
    if (cfg.repeats == 0u) throw std::invalid_argument("--repeats must be non-zero");
    if (cfg.ideal_block_size == 0u) throw std::invalid_argument("--ideal-block-size must be non-zero");
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

    if (!in) throw std::runtime_error("failed to open real-data manifest");
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string> fields = split_tab(line);
        if (fields.size() < 7u) throw std::runtime_error("real-data manifest requires 7 tab-separated columns");
        if (fields[0] == "dataset_id") continue;
        real_case entry;
        entry.dataset_id = fields[0];
        entry.matrix_path = fields[1];
        entry.rows = (std::uint32_t) std::stoul(fields[2]);
        entry.cols = (std::uint32_t) std::stoul(fields[3]);
        entry.nnz = (std::uint64_t) std::stoull(fields[4]);
        entry.slice_rows = (std::uint32_t) std::stoul(fields[5]);
        if (filter != "all" && filter != entry.dataset_id) continue;
        out.push_back(std::move(entry));
    }
    if (out.empty()) throw std::runtime_error("real-data manifest produced no cases");
    return out;
}

static std::vector<unsigned long> build_balanced_offsets(const unsigned long *row_nnz,
                                                         unsigned long slice_rows,
                                                         unsigned int part_count,
                                                         unsigned long total_rows,
                                                         unsigned long row_alignment) {
    std::vector<unsigned long> offsets((std::size_t) part_count + 2u, 0ul);
    unsigned long total_nnz = 0ul;
    unsigned long running = 0ul;
    unsigned int next_cut = 1u;

    for (unsigned long row = 0ul; row < slice_rows; ++row) total_nnz += row_nnz[row];
    for (unsigned long row = 0ul; row < slice_rows && next_cut < part_count; ++row) {
        running += row_nnz[row];
        const unsigned long target =
            (total_nnz * (unsigned long) next_cut + (unsigned long) part_count - 1ul) / (unsigned long) part_count;
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
    while (next_cut < part_count) {
        offsets[next_cut] = slice_rows;
        ++next_cut;
    }
    offsets[part_count] = slice_rows;
    offsets[part_count + 1u] = total_rows;
    return offsets;
}

static int convert_coo_part_to_compressed(const cs::sparse::coo *src,
                                          cmtx::compressed_workspace *ws,
                                          cs::sparse::compressed *dst) {
    unsigned int row = 0u;
    if (src == nullptr || ws == nullptr || dst == nullptr) return 0;
    if (src->rows > 0xffffffffu || src->cols > 0xffffffffu || src->nnz > 0xffffffffu) return 0;
    if (!cmtx::reserve(ws, (unsigned int) src->rows, (unsigned int) src->cols, (unsigned int) src->nnz)) return 0;
    if (src->nnz != 0u) {
        std::memcpy(ws->h_row_idx, src->rowIdx, (std::size_t) src->nnz * sizeof(unsigned int));
        std::memcpy(ws->h_col_idx, src->colIdx, (std::size_t) src->nnz * sizeof(unsigned int));
        std::memcpy(ws->h_in_val, src->val, (std::size_t) src->nnz * sizeof(__half));
    }
    if (!cmtx::build_pinned_triplet_to_compressed(ws,
                                                  (unsigned int) src->rows,
                                                  (unsigned int) src->cols,
                                                  (unsigned int) src->nnz,
                                                  cs::sparse::compressed_by_row)) {
        return 0;
    }
    cs::sparse::clear(dst);
    cs::sparse::init(dst,
                     (cs::types::dim_t) src->rows,
                     (cs::types::dim_t) src->cols,
                     (cs::types::nnz_t) src->nnz,
                     cs::sparse::compressed_by_row);
    if (!cs::sparse::allocate(dst)) return 0;
    for (row = 0u; row <= (unsigned int) src->rows; ++row) dst->majorPtr[row] = (cs::types::ptr_t) ws->h_major_ptr[row];
    if (src->nnz != 0u) {
        std::memcpy(dst->minorIdx, ws->h_minor_idx, (std::size_t) src->nnz * sizeof(unsigned int));
        std::memcpy(dst->val, ws->h_out_val, (std::size_t) src->nnz * sizeof(__half));
    }
    return 1;
}

static compressed_case load_real_case_as_compressed(const real_case &entry, const config &cfg) {
    cmtx::header header;
    cmtx::compressed_workspace ws;
    unsigned long *row_nnz = nullptr;
    unsigned long *part_nnz = nullptr;
    const std::size_t reader_bytes = (std::size_t) cfg.reader_bytes_mb << 20u;
    static constexpr unsigned long row_alignment = 32ul;
    compressed_case out;

    cmtx::init(&header);
    cmtx::init(&ws);
    if (!cmtx::setup(&ws, (int) cfg.device, (cudaStream_t) 0)) {
        throw std::runtime_error("failed to setup compressed workspace");
    }
    if (!cmtx::scan_row_nnz(entry.matrix_path.c_str(), &header, &row_nnz, reader_bytes)) {
        cmtx::clear(&ws);
        throw std::runtime_error("scan_row_nnz failed");
    }

    try {
        const unsigned long slice_rows = std::min<unsigned long>(entry.slice_rows, header.rows);
        const std::vector<unsigned long> row_offsets =
            build_balanced_offsets(row_nnz, slice_rows, cfg.parts, header.rows, row_alignment);
        cs::sharded<cs::sparse::coo> coo_view;

        cs::init(&coo_view);
        if (!cmtx::count_all_part_nnz(entry.matrix_path.c_str(), &header, row_offsets.data(), cfg.parts + 1u, &part_nnz)) {
            cs::clear(&coo_view);
            throw std::runtime_error("count_all_part_nnz failed");
        }
        if (!cmtx::load_part_window_coo(entry.matrix_path.c_str(),
                                        &header,
                                        row_offsets.data(),
                                        part_nnz,
                                        cfg.parts + 1u,
                                        0ul,
                                        cfg.parts,
                                        &coo_view,
                                        reader_bytes)) {
            cs::clear(&coo_view);
            throw std::runtime_error("load_part_window_coo failed");
        }

        out.parts.reserve(coo_view.num_partitions);
        out.rows = 0u;
        out.cols = (std::uint32_t) header.cols;
        out.nnz = 0u;
        for (unsigned long part = 0ul; part < coo_view.num_partitions; ++part) {
            compressed_owned owned;
            owned.part = new cs::sparse::compressed();
            cs::sparse::init(owned.part);
            if (!convert_coo_part_to_compressed(coo_view.parts[part], &ws, owned.part)) {
                cs::clear(&coo_view);
                throw std::runtime_error("convert_coo_part_to_compressed failed");
            }
            out.rows += owned.part->rows;
            out.nnz += owned.part->nnz;
            out.parts.push_back(std::move(owned));
        }
        cs::clear(&coo_view);
    } catch (...) {
        std::free(row_nnz);
        std::free(part_nnz);
        cmtx::clear(&ws);
        throw;
    }

    std::free(row_nnz);
    std::free(part_nnz);
    cmtx::clear(&ws);
    return out;
}

static int build_whole_compressed(const compressed_case &src,
                                  cs::sparse::compressed *whole,
                                  std::vector<cs::types::ptr_t> *row_ptr,
                                  std::vector<cs::types::idx_t> *col_idx,
                                  std::vector<::real::storage_t> *values) {
    std::uint64_t cursor = 0u;
    std::uint64_t global_row = 0u;
    if (whole == nullptr || row_ptr == nullptr || col_idx == nullptr || values == nullptr) return 0;

    cs::sparse::init(whole, (cs::types::dim_t) src.rows, (cs::types::dim_t) src.cols, (cs::types::nnz_t) src.nnz, cs::sparse::compressed_by_row);
    row_ptr->assign((std::size_t) src.rows + 1u, 0u);
    col_idx->assign((std::size_t) src.nnz, 0u);
    values->assign((std::size_t) src.nnz, (__half) 0);
    (*row_ptr)[0] = 0u;

    for (const compressed_owned &owned : src.parts) {
        const cs::sparse::compressed *part = owned.part;
        if (part == nullptr || part->axis != cs::sparse::compressed_by_row) return 0;
        for (cs::types::u32 row = 0u; row < part->rows; ++row) {
            const cs::types::ptr_t begin = part->majorPtr[row];
            const cs::types::ptr_t end = part->majorPtr[row + 1u];
            const cs::types::ptr_t count = end - begin;
            if (count != 0u) {
                std::memcpy(col_idx->data() + cursor, part->minorIdx + begin, (std::size_t) count * sizeof(cs::types::idx_t));
                std::memcpy(values->data() + cursor, part->val + begin, (std::size_t) count * sizeof(::real::storage_t));
            }
            cursor += count;
            (*row_ptr)[global_row + row + 1u] = (cs::types::ptr_t) cursor;
        }
        global_row += part->rows;
    }

    whole->majorPtr = row_ptr->data();
    whole->minorIdx = col_idx->data();
    whole->val = values->data();
    return 1;
}

static int contains_u32(const std::vector<unsigned int> &values, unsigned int value) {
    for (unsigned int existing : values) {
        if (existing == value) return 1;
    }
    return 0;
}

static std::vector<unsigned int> build_synthetic_row_counts(unsigned int rows,
                                                            unsigned int cols,
                                                            std::uint64_t total_nnz,
                                                            double level,
                                                            std::mt19937 *rng) {
    std::vector<unsigned int> counts((std::size_t) rows, 0u);
    const std::uint64_t base = rows == 0u ? 0u : total_nnz / rows;
    const std::uint64_t remainder = rows == 0u ? 0u : total_nnz % rows;
    const double imperfection = std::max(0.0, std::min(1.0, 1.0 - level));
    const unsigned int moves = (unsigned int) std::llround(imperfection * (double) rows * 8.0);
    const unsigned int max_delta = std::max<unsigned int>(1u, (unsigned int) std::llround((double) base * (0.25 + 1.75 * imperfection)));

    for (unsigned int row = 0u; row < rows; ++row) {
        std::uint64_t count = base + (row < remainder ? 1u : 0u);
        if (count > cols) count = cols;
        counts[(std::size_t) row] = (unsigned int) count;
    }

    for (unsigned int move = 0u; move < moves; ++move) {
        const unsigned int donor = rows != 0u ? (unsigned int) ((*rng)() % rows) : 0u;
        const unsigned int receiver = rows != 0u ? (unsigned int) ((*rng)() % rows) : 0u;
        const unsigned int requested = 1u + (unsigned int) ((*rng)() % max_delta);
        const unsigned int donor_slack = counts[(std::size_t) donor] > 1u ? counts[(std::size_t) donor] - 1u : 0u;
        const unsigned int receiver_cap = counts[(std::size_t) receiver] < cols ? cols - counts[(std::size_t) receiver] : 0u;
        const unsigned int delta = std::min(requested, std::min(donor_slack, receiver_cap));
        if (delta == 0u || donor == receiver) continue;
        counts[(std::size_t) donor] -= delta;
        counts[(std::size_t) receiver] += delta;
    }

    return counts;
}

static void pick_unique_local_positions(unsigned int valid_cols,
                                        unsigned int take,
                                        std::mt19937 *rng,
                                        std::vector<unsigned int> *out) {
    std::vector<unsigned int> positions(valid_cols);
    std::iota(positions.begin(), positions.end(), 0u);
    std::shuffle(positions.begin(), positions.end(), *rng);
    positions.resize(take);
    std::sort(positions.begin(), positions.end());
    *out = std::move(positions);
}

static void add_random_blocks_excluding(unsigned int col_blocks,
                                        unsigned int template_begin,
                                        unsigned int template_width,
                                        unsigned int target_total,
                                        std::mt19937 *rng,
                                        std::vector<unsigned int> *block_cols) {
    unsigned int attempts = 0u;
    while (block_cols->size() < target_total && attempts < col_blocks * 8u + 64u) {
        const unsigned int block = col_blocks != 0u ? ((*rng)() % col_blocks) : 0u;
        const int in_template = block >= template_begin && block < template_begin + template_width;
        ++attempts;
        if (in_template || contains_u32(*block_cols, block)) continue;
        block_cols->push_back(block);
    }
    for (unsigned int block = 0u; block < col_blocks && block_cols->size() < target_total; ++block) {
        const int in_template = block >= template_begin && block < template_begin + template_width;
        if (in_template || contains_u32(*block_cols, block)) continue;
        block_cols->push_back(block);
    }
}

static compressed_owned make_synthetic_part(const cs::sparse::compressed *reference,
                                            unsigned int ideal_block_size,
                                            double level,
                                            std::uint64_t seed) {
    std::mt19937 rng((std::mt19937::result_type) seed);
    compressed_owned owned;
    const unsigned int rows = reference->rows;
    const unsigned int cols = reference->cols;
    const unsigned int block_size = std::max(1u, ideal_block_size);
    const unsigned int col_blocks = (cols + block_size - 1u) / block_size;
    const std::vector<unsigned int> row_counts =
        build_synthetic_row_counts(rows, cols, reference->nnz, level, &rng);
    std::size_t cursor = 0u;

    owned.part = new cs::sparse::compressed();
    cs::sparse::init(owned.part, rows, cols, reference->nnz, cs::sparse::compressed_by_row);
    if (!cs::sparse::allocate(owned.part)) throw std::runtime_error("failed to allocate synthetic compressed part");

    owned.part->majorPtr[0] = 0u;
    for (unsigned int row = 0u; row < rows; ++row) {
        owned.part->majorPtr[row + 1u] = owned.part->majorPtr[row] + row_counts[(std::size_t) row];
    }

    for (unsigned int row_block = 0u; row_block < rows; row_block += block_size) {
        const unsigned int row_end = std::min<unsigned int>(rows, row_block + block_size);
        const double imperfection = std::max(0.0, std::min(1.0, 1.0 - level));
        const double degrade_roll = (double) rng() / (double) rng.max();
        const double degraded_block_fraction = std::max(0.0, std::min(0.22, imperfection * (0.16 + 0.24 * imperfection)));
        const int degraded_block = degrade_roll < degraded_block_fraction;
        const double block_severity =
            degraded_block ? std::min(1.0, imperfection * (0.75 + 0.25 * ((double) rng() / (double) rng.max())))
                           : imperfection * 0.20 * ((double) rng() / (double) rng.max());
        unsigned int template_width = 1u;
        unsigned int template_begin = 0u;
        std::vector<unsigned int> template_cols;

        for (unsigned int row = row_block; row < row_end; ++row) {
            const unsigned int min_blocks =
                (row_counts[(std::size_t) row] + block_size - 1u) / block_size;
            template_width = std::max(template_width, min_blocks);
        }
        if (imperfection > 0.0 && col_blocks > template_width) {
            const unsigned int max_extra =
                std::min<unsigned int>(col_blocks - template_width,
                                       std::max<unsigned int>(
                                           degraded_block ? 2u : 1u,
                                           (unsigned int) std::llround((double) template_width
                                                                       * (degraded_block ? (1.0 + 5.0 * block_severity)
                                                                                         : (0.25 + 1.25 * imperfection)))));
            const double extra_scale =
                degraded_block ? (0.60 + 0.40 * ((double) rng() / (double) rng.max())) : ((double) rng() / (double) rng.max()) * 0.35;
            const unsigned int extra_blocks =
                (unsigned int) std::llround((double) max_extra * extra_scale * std::max(imperfection, block_severity));
            template_width = std::min<unsigned int>(col_blocks, template_width + extra_blocks);
        }
        if (template_width > col_blocks) template_width = col_blocks;
        if (col_blocks > template_width) {
            template_begin = (unsigned int) (rng() % (col_blocks - template_width + 1u));
        }
        template_cols.resize(template_width);
        for (unsigned int i = 0u; i < template_width; ++i) template_cols[i] = template_begin + i;

        for (unsigned int row = row_block; row < row_end; ++row) {
            const unsigned int count = row_counts[(std::size_t) row];
            const double row_roll = (double) rng() / (double) rng.max();
            const int degraded_row =
                degraded_block && row_roll < std::min(0.80, 0.18 + 0.22 * block_severity + 0.08 * imperfection);
            const double shared_fraction =
                imperfection == 0.0 ? 1.0
                                    : (degraded_row ? std::max(0.12, std::min(0.92, 0.90 - 1.10 * block_severity + 0.04 * row_roll))
                                                    : std::max(0.86, std::min(1.0, 1.0 - 0.10 * imperfection - 0.03 * row_roll)));
            const double density =
                imperfection == 0.0 ? 1.0
                                    : (degraded_row ? std::max(0.08, std::min(0.95, 0.94 - 1.35 * block_severity + 0.04 * row_roll))
                                                    : std::max(0.90, std::min(1.0, 1.0 - 0.06 * imperfection - 0.02 * row_roll)));
            const unsigned int target_values_per_block =
                std::max(1u, std::min(block_size, (unsigned int) std::llround(density * (double) block_size)));
            unsigned int blocks_needed = count == 0u ? 0u : (count + target_values_per_block - 1u) / target_values_per_block;
            const unsigned int min_blocks = count == 0u ? 0u : (count + block_size - 1u) / block_size;
            unsigned int shared_blocks = 0u;
            std::vector<unsigned int> block_cols;
            std::size_t row_cursor = cursor;

            if (blocks_needed < min_blocks) blocks_needed = min_blocks;
            if (degraded_row && blocks_needed != 0u) {
                const unsigned int span_cap = std::max<unsigned int>(min_blocks, template_width);
                const unsigned int target_span =
                    std::min<unsigned int>(col_blocks,
                                           min_blocks +
                                               std::max<unsigned int>(
                                                   1u,
                                                   (unsigned int) std::llround((double) (span_cap - min_blocks)
                                                                               * std::min(1.0, 0.10 + 0.80 * block_severity + 0.10 * row_roll))));
                if (target_span > blocks_needed) blocks_needed = target_span;
            }
            if (blocks_needed > col_blocks) blocks_needed = col_blocks;
            if (blocks_needed != 0u) {
                shared_blocks = (unsigned int) std::floor((double) blocks_needed * shared_fraction + 1.0e-9);
                if (shared_fraction > 0.0 && shared_blocks == 0u) shared_blocks = 1u;
                if (shared_blocks > template_width) shared_blocks = template_width;
                if (shared_blocks > blocks_needed) shared_blocks = blocks_needed;
            }

            for (unsigned int i = 0u; i < shared_blocks; ++i) block_cols.push_back(template_cols[i]);
            add_random_blocks_excluding(col_blocks, template_begin, template_width, blocks_needed, &rng, &block_cols);
            if (block_cols.size() < blocks_needed) {
                for (unsigned int i = 0u; i < template_width && block_cols.size() < blocks_needed; ++i) {
                    if (!contains_u32(block_cols, template_cols[i])) block_cols.push_back(template_cols[i]);
                }
            }
            std::sort(block_cols.begin(), block_cols.end());

            while (!block_cols.empty()) {
                unsigned int total_capacity = 0u;
                for (unsigned int block_col : block_cols) {
                    const unsigned int col_begin = block_col * block_size;
                    const unsigned int valid_cols = std::min<unsigned int>(block_size, cols - col_begin);
                    total_capacity += std::min<unsigned int>(valid_cols, target_values_per_block);
                }
                if (total_capacity >= count || block_cols.size() == col_blocks) break;
                add_random_blocks_excluding(col_blocks, template_begin, template_width, (unsigned int) block_cols.size() + 1u, &rng, &block_cols);
                std::sort(block_cols.begin(), block_cols.end());
            }

            std::vector<unsigned int> per_block_take(block_cols.size(), 0u);
            unsigned int remaining = count;
            while (remaining != 0u && !block_cols.empty()) {
                int progressed = 0;
                for (unsigned int idx = 0u; idx < block_cols.size() && remaining != 0u; ++idx) {
                    const unsigned int col_begin = block_cols[idx] * block_size;
                    const unsigned int valid_cols = std::min<unsigned int>(block_size, cols - col_begin);
                    const unsigned int per_block_cap = std::min<unsigned int>(valid_cols, target_values_per_block);
                    if (per_block_take[idx] >= per_block_cap) continue;
                    ++per_block_take[idx];
                    --remaining;
                    progressed = 1;
                }
                if (!progressed) break;
            }

            for (unsigned int idx = 0u; idx < block_cols.size(); ++idx) {
                const unsigned int block_col = block_cols[idx];
                const unsigned int take = per_block_take[idx];
                const unsigned int col_begin = block_col * block_size;
                const unsigned int valid_cols = std::min<unsigned int>(block_size, cols - col_begin);
                std::vector<unsigned int> local_positions;
                if (take == 0u) continue;
                pick_unique_local_positions(valid_cols, take, &rng, &local_positions);
                for (unsigned int local : local_positions) {
                    owned.part->minorIdx[row_cursor] = col_begin + local;
                    owned.part->val[row_cursor] =
                        __float2half(1.0f + (float) ((seed + row * 17u + row_cursor) % 29u));
                    ++row_cursor;
                }
            }

            if (row_cursor != cursor + count) {
                throw std::runtime_error("synthetic row generation failed to match reference nnz");
            }
            std::sort(owned.part->minorIdx + cursor, owned.part->minorIdx + row_cursor);
            cursor = row_cursor;
        }
    }

    return owned;
}

static compressed_case make_synthetic_case(const compressed_case &reference,
                                          unsigned int ideal_block_size,
                                          double level,
                                          unsigned int seed) {
    compressed_case out;
    out.rows = reference.rows;
    out.cols = reference.cols;
    out.nnz = reference.nnz;
    out.parts.reserve(reference.parts.size());
    for (std::size_t i = 0u; i < reference.parts.size(); ++i) {
        const std::uint64_t part_seed =
            (std::uint64_t) seed + 0x9e3779b97f4a7c15ull * (std::uint64_t) (i + 1u) + (std::uint64_t) std::llround(level * 1000.0);
        out.parts.push_back(make_synthetic_part(reference.parts[i].part, ideal_block_size, level, part_seed));
    }
    return out;
}

static build_metrics evaluate_build_case(const compressed_case &input,
                                         const std::string &source,
                                         const std::string &dataset_id,
                                         const std::string &level_label,
                                         const config &cfg) {
    cs::sparse::compressed whole;
    std::vector<cs::types::ptr_t> row_ptr;
    std::vector<cs::types::idx_t> col_idx;
    std::vector<::real::storage_t> values;
    std::vector<blocked_ell_owned> built_parts;
    build_metrics out;
    csc::blocked_ell_tune_result tune = {};
    double total_ms = 0.0;

    cs::sparse::init(&whole);
    if (!build_whole_compressed(input, &whole, &row_ptr, &col_idx, &values)) {
        throw std::runtime_error("build_whole_compressed failed");
    }

    for (unsigned int repeat = 0u; repeat < cfg.repeats; ++repeat) {
        csc::blocked_ell_tune_result repeat_tune = {};
        std::vector<blocked_ell_owned> repeat_parts;
        const auto start = std::chrono::steady_clock::now();

        if (!csc::choose_blocked_ell_block_size(&whole, cfg.candidates.data(), (unsigned int) cfg.candidates.size(), &repeat_tune)) {
            throw std::runtime_error("choose_blocked_ell_block_size failed");
        }
        repeat_parts.reserve(input.parts.size());
        for (const compressed_owned &owned : input.parts) {
            blocked_ell_owned blocked;
            blocked.part = new cs::sparse::blocked_ell();
            cs::sparse::init(blocked.part);
            if (!csc::blocked_ell_from_compressed(owned.part, repeat_tune.block_size, blocked.part)) {
                throw std::runtime_error("blocked_ell_from_compressed failed");
            }
            repeat_parts.push_back(std::move(blocked));
        }

        total_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        if (repeat == 0u) {
            tune = repeat_tune;
            built_parts = std::move(repeat_parts);
        }
    }

    out.source = source;
    out.dataset_id = dataset_id;
    out.level_label = level_label;
    out.rows = input.rows;
    out.cols = input.cols;
    out.nnz = input.nnz;
    out.parts = (std::uint32_t) input.parts.size();
    out.ideal_block_size = cfg.ideal_block_size;
    out.tuned_block_size = tune.block_size;
    out.tuned_fill_ratio = tune.fill_ratio;
    out.tuned_padded_bytes = tune.padded_bytes;
    out.build_ms_total = total_ms / (double) cfg.repeats;
    out.build_ms_per_part = built_parts.empty() ? 0.0 : out.build_ms_total / (double) built_parts.size();

    {
        std::size_t padded_values_bytes = 0u;
        std::size_t active_slots = 0u;
        std::size_t slot_capacity = 0u;

        for (const blocked_ell_owned &owned : built_parts) {
            const cs::sparse::blocked_ell *part = owned.part;
            const std::size_t row_blocks = (std::size_t) cs::sparse::row_block_count(part);
            const std::size_t ell_width = (std::size_t) cs::sparse::ell_width_blocks(part);
            const std::size_t idx_count = row_blocks * ell_width;
            padded_values_bytes += (std::size_t) part->rows * (std::size_t) part->ell_cols * sizeof(::real::storage_t);
            slot_capacity += idx_count;
            for (std::size_t idx = 0u; idx < idx_count; ++idx) {
                if (part->blockColIdx[idx] != cs::sparse::blocked_ell_invalid_col) ++active_slots;
            }
        }

        out.padded_bytes = padded_values_bytes;
        out.weighted_fill_ratio = slot_capacity == 0u ? 1.0 : (double) active_slots / (double) slot_capacity;
        out.storage_amplification =
            out.nnz == 0u ? 0.0 : (double) padded_values_bytes / ((double) out.nnz * (double) sizeof(::real::storage_t));
    }

    return out;
}

static void print_case_result(const build_metrics &result) {
    std::printf("blocked_ell_opt: source=%s dataset=%s level=%s rows=%llu cols=%u nnz=%llu parts=%u ideal_block_size=%u tuned_block_size=%u tuned_fill_ratio=%.6f weighted_fill_ratio=%.6f tuned_padded_bytes=%zu padded_bytes=%zu storage_amplification=%.6f build_ms_total=%.3f build_ms_per_part=%.3f\n",
                result.source.c_str(),
                result.dataset_id.c_str(),
                result.level_label.c_str(),
                (unsigned long long) result.rows,
                result.cols,
                (unsigned long long) result.nnz,
                result.parts,
                result.ideal_block_size,
                result.tuned_block_size,
                result.tuned_fill_ratio,
                result.weighted_fill_ratio,
                result.tuned_padded_bytes,
                result.padded_bytes,
                result.storage_amplification,
                result.build_ms_total,
                result.build_ms_per_part);
}

static void print_case_delta(const build_metrics &real_result, const build_metrics &synthetic_result) {
    std::printf("blocked_ell_opt_delta: dataset=%s level=%s tuned_block_size_delta=%d tuned_fill_ratio_delta=%.6f weighted_fill_ratio_delta=%.6f padded_bytes_delta=%lld storage_amplification_delta=%.6f build_ms_total_delta=%.3f\n",
                real_result.dataset_id.c_str(),
                synthetic_result.level_label.c_str(),
                (int) synthetic_result.tuned_block_size - (int) real_result.tuned_block_size,
                synthetic_result.tuned_fill_ratio - real_result.tuned_fill_ratio,
                synthetic_result.weighted_fill_ratio - real_result.weighted_fill_ratio,
                (long long) synthetic_result.padded_bytes - (long long) real_result.padded_bytes,
                synthetic_result.storage_amplification - real_result.storage_amplification,
                synthetic_result.build_ms_total - real_result.build_ms_total);
}

static void print_config(const config &cfg) {
    std::printf("config: manifest=%s dataset=%s device=%u parts=%u reader_bytes_mb=%u repeats=%u seed=%u ideal_block_size=%u candidates=",
                cfg.manifest_path.c_str(),
                cfg.dataset_filter.c_str(),
                cfg.device,
                cfg.parts,
                cfg.reader_bytes_mb,
                cfg.repeats,
                cfg.seed,
                cfg.ideal_block_size);
    for (std::size_t i = 0u; i < cfg.candidates.size(); ++i) {
        std::printf("%s%u", i == 0u ? "" : ",", cfg.candidates[i]);
    }
    std::printf(" levels=");
    for (std::size_t i = 0u; i < cfg.levels.size(); ++i) {
        std::printf("%s%.2f", i == 0u ? "" : ",", cfg.levels[i]);
    }
    std::printf("\n");
}

static std::string synthetic_level_label(double level) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.2f", level);
    return std::string(buffer);
}

} // namespace

int main(int argc, char **argv) {
    try {
        cellerator::bench::benchmark_mutex_guard benchmark_mutex("blockedEllOptimizationBench");
        const config cfg = parse_args(argc, argv);
        int device_count = 0;
        if (!check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount")) return 1;
        if (device_count <= 0 || cfg.device >= (unsigned int) device_count) {
            std::fprintf(stderr, "requested CUDA device %u is unavailable\n", cfg.device);
            return 1;
        }

        print_config(cfg);
        const std::vector<real_case> cases = load_manifest(cfg.manifest_path, cfg.dataset_filter);
        for (const real_case &entry : cases) {
            const compressed_case real_input = load_real_case_as_compressed(entry, cfg);
            const build_metrics real_result = evaluate_build_case(real_input, "real", entry.dataset_id, "real", cfg);
            print_case_result(real_result);

            for (double level : cfg.levels) {
                const compressed_case synthetic_input =
                    make_synthetic_case(real_input, cfg.ideal_block_size, level, cfg.seed);
                const build_metrics synthetic_result = evaluate_build_case(synthetic_input,
                                                                           "synthetic",
                                                                           entry.dataset_id,
                                                                           synthetic_level_label(level),
                                                                           cfg);
                print_case_result(synthetic_result);
                print_case_delta(real_result, synthetic_result);
            }
        }
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "blockedEllOptimizationBench: %s\n", e.what());
        return 1;
    }
}
