#include "../src/compute/autograd/autograd.hh"
#include "../src/ingest/mtx/mtx_reader.cuh"
#include "../src/ingest/series/series_ingest.cuh"
#include "benchmark_mutex.hh"

#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace autograd = ::cellerator::compute::autograd;
namespace cmtx = ::cellerator::ingest::mtx;
namespace cseries = ::cellerator::ingest::series;
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

struct bench_config {
    std::string manifest_path = "bench/real_data/embryo_spmm_manifest.tsv";
    std::string dataset_filter = "all";
    std::uint32_t warmup = 3u;
    std::uint32_t iters = 10u;
    std::uint32_t reader_bytes_mb = 16u;
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
    compressed_owned(compressed_owned &&other) noexcept : part(other.part) { other.part = nullptr; }
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
    blocked_ell_owned(blocked_ell_owned &&other) noexcept : part(other.part) { other.part = nullptr; }
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

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

std::uint32_t parse_u32(const char *value, const char *label) {
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (value == nullptr || *value == '\0' || end == nullptr || *end != '\0' || parsed > 0xfffffffful) {
        throw std::invalid_argument(std::string("invalid integer for ") + label);
    }
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
        if (arg == "--manifest") {
            cfg.manifest_path = require_value("--manifest");
        } else if (arg == "--dataset") {
            cfg.dataset_filter = require_value("--dataset");
        } else if (arg == "--warmup") {
            cfg.warmup = parse_u32(require_value("--warmup"), "--warmup");
        } else if (arg == "--iters") {
            cfg.iters = parse_u32(require_value("--iters"), "--iters");
        } else if (arg == "--reader-bytes-mb") {
            cfg.reader_bytes_mb = parse_u32(require_value("--reader-bytes-mb"), "--reader-bytes-mb");
        } else if (arg == "-h" || arg == "--help") {
            std::cout
                << "Usage: cellShardRealSpmmBench [--manifest path] [--dataset id|all] [--warmup N] [--iters N] [--reader-bytes-mb N]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument(std::string("unknown argument: ") + arg);
        }
    }
    return cfg;
}

std::vector<std::string> split_tab(const std::string &line) {
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

std::vector<real_case> load_manifest(const std::string &path, const std::string &filter) {
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
        entry.rhs_cols = (std::uint32_t) std::stoul(fields[6]);
        if (filter != "all" && filter != entry.dataset_id) continue;
        out.push_back(std::move(entry));
    }
    if (out.empty()) throw std::runtime_error("real-data manifest produced no cases");
    return out;
}

std::vector<unsigned long> build_balanced_offsets(const unsigned long *row_nnz,
                                                  unsigned long slice_rows,
                                                  unsigned int shard_count,
                                                  unsigned long total_rows,
                                                  unsigned long row_alignment) {
    std::vector<unsigned long> offsets((std::size_t) shard_count + 2u, 0ul);
    unsigned long total_nnz = 0ul;
    unsigned long running = 0ul;
    unsigned int next_cut = 1u;

    for (unsigned long row = 0; row < slice_rows; ++row) total_nnz += row_nnz[row];
    for (unsigned long row = 0; row < slice_rows && next_cut < shard_count; ++row) {
        running += row_nnz[row];
        const unsigned long target = (total_nnz * (unsigned long) next_cut + (unsigned long) shard_count - 1ul) / (unsigned long) shard_count;
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

std::vector<compressed_owned> convert_window_parts_to_csr(const cs::sharded<cs::sparse::coo> &coo_view,
                                                          cmtx::compressed_workspace *ws) {
    std::vector<compressed_owned> out;
    out.reserve(coo_view.num_parts);
    for (unsigned long part = 0; part < coo_view.num_parts; ++part) {
        compressed_owned owned;
        owned.part = new cs::sparse::compressed();
        cs::sparse::init(owned.part);
        if (!cseries::convert_coo_part_to_csr(coo_view.parts[part], ws, owned.part)) {
            throw std::runtime_error("convert_coo_part_to_csr failed");
        }
        out.push_back(std::move(owned));
    }
    return out;
}

std::vector<blocked_ell_owned> convert_window_parts_to_blocked_ell(const std::vector<compressed_owned> &compressed_parts,
                                                                   double *mean_fill_ratio) {
    static constexpr unsigned int candidates[] = {8u, 16u, 32u};
    std::vector<blocked_ell_owned> out;
    double fill_sum = 0.0;
    out.reserve(compressed_parts.size());
    for (const compressed_owned &compressed : compressed_parts) {
        cs::convert::blocked_ell_tune_result tune = {};
        blocked_ell_owned owned;
        owned.part = new cs::sparse::blocked_ell();
        cs::sparse::init(owned.part);
        if (!cs::convert::blocked_ell_from_compressed_auto(compressed.part, candidates, 3u, owned.part, &tune)) {
            throw std::runtime_error("blocked_ell_from_compressed_auto failed");
        }
        fill_sum += tune.fill_ratio;
        out.push_back(std::move(owned));
    }
    if (mean_fill_ratio != nullptr) {
        *mean_fill_ratio = out.empty() ? 0.0 : (fill_sum / (double) out.size());
    }
    return out;
}

std::unique_ptr<float[]> make_host_rhs_f32(std::uint32_t logical_rows,
                                           std::uint32_t padded_rows,
                                           std::uint32_t cols) {
    auto out = std::make_unique<float[]>(static_cast<std::size_t>(padded_rows) * cols);
    std::fill_n(out.get(), static_cast<std::size_t>(padded_rows) * cols, 0.0f);
    for (std::uint32_t r = 0; r < logical_rows; ++r) {
        for (std::uint32_t c = 0; c < cols; ++c) {
            out[(std::size_t) r * cols + c] = 0.0625f + static_cast<float>((r * 17u + c) % 23u) * 0.03125f;
        }
    }
    return out;
}

std::unique_ptr<__half[]> make_host_rhs_f16(std::uint32_t logical_rows,
                                            std::uint32_t padded_rows,
                                            std::uint32_t cols) {
    auto out = std::make_unique<__half[]>(static_cast<std::size_t>(padded_rows) * cols);
    std::fill_n(out.get(), static_cast<std::size_t>(padded_rows) * cols, __float2half(0.0f));
    for (std::uint32_t r = 0; r < logical_rows; ++r) {
        for (std::uint32_t c = 0; c < cols; ++c) {
            out[(std::size_t) r * cols + c] = __float2half(0.0625f + static_cast<float>((r * 17u + c) % 23u) * 0.03125f);
        }
    }
    return out;
}

template<typename Fn>
double time_loop(std::uint32_t warmup, std::uint32_t iters, Fn &&fn) {
    for (std::uint32_t i = 0; i < warmup; ++i) fn();
    const auto start = std::chrono::steady_clock::now();
    for (std::uint32_t i = 0; i < iters; ++i) fn();
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

void print_result(const real_case &entry,
                  const char *layout,
                  unsigned int slot_count,
                  std::uint32_t slice_rows,
                  std::uint32_t rhs_cols,
                  double convert_ms,
                  double fill_ratio,
                  double avg_ms,
                  std::uint64_t nnz) {
    const double nnz_per_s = avg_ms > 0.0 ? (double) nnz / (avg_ms * 1.0e-3) : 0.0;
    const double out_el_per_s = avg_ms > 0.0 ? ((double) slice_rows * rhs_cols) / (avg_ms * 1.0e-3) : 0.0;
    std::cout
        << "dataset=" << entry.dataset_id
        << " layout=" << layout
        << " slots=" << slot_count
        << " slice_rows=" << slice_rows
        << " cols=" << entry.cols
        << " rhs_cols=" << rhs_cols
        << " nnz=" << nnz
        << " convert_ms=" << convert_ms
        << " fill_ratio=" << fill_ratio
        << " avg_ms=" << avg_ms
        << " nnz_per_s=" << nnz_per_s
        << " out_el_per_s=" << out_el_per_s
        << '\n';
}

void benchmark_compressed(const real_case &entry,
                          autograd::fleet_context *fleet,
                          const unsigned int *slots,
                          unsigned int slot_count,
                          const std::vector<compressed_owned> &parts,
                          std::uint32_t rhs_cols,
                          std::uint32_t warmup,
                          std::uint32_t iters,
                          std::uint64_t total_nnz) {
    std::vector<autograd::device_buffer<std::uint32_t>> major(slot_count);
    std::vector<autograd::device_buffer<std::uint32_t>> minor(slot_count);
    std::vector<autograd::device_buffer<__half>> values(slot_count);
    std::vector<autograd::device_buffer<float>> rhs(slot_count);
    std::vector<autograd::device_buffer<float>> out(slot_count);
    std::vector<const std::uint32_t *> major_ptr(slot_count);
    std::vector<const std::uint32_t *> minor_ptr(slot_count);
    std::vector<const __half *> value_ptr(slot_count);
    std::vector<const float *> rhs_ptr(slot_count);
    std::vector<float *> out_ptr(slot_count);
    std::vector<std::uint32_t> rows(slot_count);
    std::vector<std::uint32_t> cols(slot_count);
    std::vector<std::int64_t> rhs_ld(slot_count, (std::int64_t) rhs_cols);
    std::vector<std::int64_t> out_cols(slot_count, (std::int64_t) rhs_cols);
    std::vector<std::int64_t> out_ld(slot_count, (std::int64_t) rhs_cols);
    auto rhs_host = make_host_rhs_f32(entry.cols, entry.cols, rhs_cols);
    std::uint32_t total_rows = 0u;

    for (unsigned int i = 0; i < slot_count; ++i) {
        const int device = autograd::fleet_device_id(*fleet, slots[i]);
        const cs::sparse::compressed *part = parts[i].part;
        rows[i] = part->rows;
        total_rows += rows[i];
        cols[i] = part->cols;
        major[i] = allocate_on_device<std::uint32_t>(device, (std::size_t) part->rows + 1u);
        minor[i] = allocate_on_device<std::uint32_t>(device, part->nnz);
        values[i] = allocate_on_device<__half>(device, part->nnz);
        rhs[i] = allocate_on_device<float>(device, (std::size_t) entry.cols * rhs_cols);
        out[i] = allocate_on_device<float>(device, (std::size_t) part->rows * rhs_cols);
        upload_on_device(device, &major[i], part->majorPtr, (std::size_t) part->rows + 1u);
        upload_on_device(device, &minor[i], part->minorIdx, part->nnz);
        upload_on_device(device, &values[i], part->val, part->nnz);
        upload_on_device(device, &rhs[i], rhs_host.get(), (std::size_t) entry.cols * rhs_cols);
        major_ptr[i] = major[i].data;
        minor_ptr[i] = minor[i].data;
        value_ptr[i] = values[i].data;
        rhs_ptr[i] = rhs[i].data;
        out_ptr[i] = out[i].data;
    }

    autograd::synchronize_slots(*fleet, slots, slot_count);
    const double total_ms = time_loop(warmup, iters, [&]() {
        autograd::dist::launch_csr_spmm_fwd_f16_f32(
            fleet,
            slots,
            slot_count,
            major_ptr.data(),
            minor_ptr.data(),
            value_ptr.data(),
            rows.data(),
            cols.data(),
            rhs_ptr.data(),
            rhs_ld.data(),
            out_cols.data(),
            out_ptr.data(),
            out_ld.data());
        autograd::synchronize_slots(*fleet, slots, slot_count);
    });

    print_result(entry, "compressed", slot_count, total_rows, rhs_cols, 0.0, 0.0, total_ms / (double) iters, total_nnz);
}

void benchmark_blocked_ell(const real_case &entry,
                           autograd::fleet_context *fleet,
                           const unsigned int *slots,
                           unsigned int slot_count,
                           const std::vector<compressed_owned> &compressed_parts,
                           std::uint32_t rhs_cols,
                           std::uint32_t warmup,
                           std::uint32_t iters,
                           std::uint64_t total_nnz) {
    std::vector<blocked_ell_owned> parts;
    std::vector<autograd::cusparse_cache> cache(slot_count);
    std::vector<autograd::device_buffer<std::uint32_t>> block_idx(slot_count);
    std::vector<autograd::device_buffer<__half>> values(slot_count);
    std::vector<autograd::device_buffer<__half>> rhs(slot_count);
    std::vector<autograd::device_buffer<float>> out(slot_count);
    std::vector<const void *> matrix_token(slot_count);
    std::vector<const std::uint32_t *> block_idx_ptr(slot_count);
    std::vector<const __half *> value_ptr(slot_count);
    std::vector<const __half *> rhs_ptr(slot_count);
    std::vector<float *> out_ptr(slot_count);
    std::vector<std::uint32_t> rows(slot_count);
    std::vector<std::uint32_t> cols(slot_count);
    std::vector<std::uint32_t> padded_cols(slot_count);
    std::vector<std::uint32_t> block_size(slot_count);
    std::vector<std::uint32_t> ell_cols(slot_count);
    std::vector<std::int64_t> rhs_ld(slot_count, (std::int64_t) rhs_cols);
    std::vector<std::int64_t> out_cols(slot_count, (std::int64_t) rhs_cols);
    std::vector<std::int64_t> out_ld(slot_count, (std::int64_t) rhs_cols);
    double fill_ratio = 0.0;
    const auto convert_start = std::chrono::steady_clock::now();
    std::uint32_t total_rows = 0u;

    parts = convert_window_parts_to_blocked_ell(compressed_parts, &fill_ratio);
    const auto convert_stop = std::chrono::steady_clock::now();
    const double convert_ms = std::chrono::duration<double, std::milli>(convert_stop - convert_start).count();

    for (unsigned int i = 0; i < slot_count; ++i) {
        const int device = autograd::fleet_device_id(*fleet, slots[i]);
        const cs::sparse::blocked_ell *part = parts[i].part;
        const std::size_t row_blocks = cs::sparse::row_block_count(part);
        autograd::init(cache.data() + i);
        rows[i] = part->rows;
        total_rows += rows[i];
        block_size[i] = part->block_size;
        ell_cols[i] = part->ell_cols;
        padded_cols[i] = block_size[i] == 0u
            ? part->cols
            : ((part->cols + block_size[i] - 1u) / block_size[i]) * block_size[i];
        cols[i] = padded_cols[i];
        auto rhs_host = make_host_rhs_f16(part->cols, padded_cols[i], rhs_cols);
        block_idx[i] = allocate_on_device<std::uint32_t>(device, row_blocks * cs::sparse::ell_width_blocks(part));
        values[i] = allocate_on_device<__half>(device, (std::size_t) part->rows * part->ell_cols);
        rhs[i] = allocate_on_device<__half>(device, (std::size_t) padded_cols[i] * rhs_cols);
        out[i] = allocate_on_device<float>(device, (std::size_t) part->rows * rhs_cols);
        upload_on_device(device, &block_idx[i], part->blockColIdx, row_blocks * cs::sparse::ell_width_blocks(part));
        upload_on_device(device, &values[i], part->val, (std::size_t) part->rows * part->ell_cols);
        upload_on_device(device, &rhs[i], rhs_host.get(), (std::size_t) padded_cols[i] * rhs_cols);
        matrix_token[i] = part->val;
        block_idx_ptr[i] = block_idx[i].data;
        value_ptr[i] = values[i].data;
        rhs_ptr[i] = rhs[i].data;
        out_ptr[i] = out[i].data;
    }

    autograd::synchronize_slots(*fleet, slots, slot_count);
    const double total_ms = time_loop(warmup, iters, [&]() {
        autograd::dist::launch_blocked_ell_spmm_fwd_f16_f16_f32_lib(
            fleet,
            cache.data(),
            slots,
            slot_count,
            matrix_token.data(),
            block_idx_ptr.data(),
            value_ptr.data(),
            rows.data(),
            cols.data(),
            block_size.data(),
            ell_cols.data(),
            rhs_ptr.data(),
            rhs_ld.data(),
            out_cols.data(),
            out_ptr.data(),
            out_ld.data());
        autograd::synchronize_slots(*fleet, slots, slot_count);
    });

    for (unsigned int i = 0; i < slot_count; ++i) autograd::clear(cache.data() + i);
    print_result(entry, "blocked_ell", slot_count, total_rows, rhs_cols, convert_ms, fill_ratio, total_ms / (double) iters, total_nnz);
}

void run_case(const real_case &entry, const bench_config &cfg, autograd::fleet_context *fleet) {
    cmtx::header header;
    unsigned long *row_nnz = nullptr;
    unsigned long *part_nnz = nullptr;
    const std::size_t reader_bytes = (std::size_t) cfg.reader_bytes_mb << 20u;
    static constexpr unsigned long kBlockedEllRowAlignment = 32ul;
    cmtx::compressed_workspace ws;

    cmtx::init(&header);
    cmtx::init(&ws);
    if (!cmtx::setup(&ws, 0, (cudaStream_t) 0)) throw std::runtime_error("failed to setup compressed workspace");
    if (!cmtx::scan_row_nnz(entry.matrix_path.c_str(), &header, &row_nnz, reader_bytes)) throw std::runtime_error("scan_row_nnz failed");
    const unsigned long slice_rows = std::min<unsigned long>(entry.slice_rows, header.rows);

    for (unsigned int slot_count : {2u, 4u}) {
        if (fleet->local.device_count < slot_count) continue;
        std::vector<unsigned long> row_offsets =
            build_balanced_offsets(row_nnz, slice_rows, slot_count, header.rows, kBlockedEllRowAlignment);
        cs::sharded<cs::sparse::coo> coo_view;
        std::vector<compressed_owned> compressed_parts;
        unsigned int slots[4] = {};

        cs::init(&coo_view);
        if (slot_count == 2u) require(autograd::default_pair_slots(0u, slots, 4u) == 2u, "pair slots unavailable");
        else require(autograd::default_fleet_slots(slots, 4u) == 4u, "fleet slots unavailable");
        if (!cmtx::count_all_part_nnz(entry.matrix_path.c_str(), &header, row_offsets.data(), slot_count + 1u, &part_nnz)) {
            cs::clear(&coo_view);
            throw std::runtime_error("count_all_part_nnz failed");
        }
        if (!cmtx::load_part_window_coo(entry.matrix_path.c_str(),
                                        &header,
                                        row_offsets.data(),
                                        part_nnz,
                                        slot_count + 1u,
                                        0ul,
                                        slot_count,
                                        &coo_view,
                                        reader_bytes)) {
            std::free(part_nnz);
            cs::clear(&coo_view);
            throw std::runtime_error("load_part_window_coo failed");
        }
        compressed_parts = convert_window_parts_to_csr(coo_view, &ws);
        std::uint64_t total_nnz = 0u;
        for (const compressed_owned &part : compressed_parts) total_nnz += part.part->nnz;
        benchmark_compressed(entry, fleet, slots, slot_count, compressed_parts, entry.rhs_cols, cfg.warmup, cfg.iters, total_nnz);
        benchmark_blocked_ell(entry, fleet, slots, slot_count, compressed_parts, entry.rhs_cols, cfg.warmup, cfg.iters, total_nnz);
        std::free(part_nnz);
        part_nnz = nullptr;
        cs::clear(&coo_view);
    }

    cmtx::clear(&ws);
    std::free(row_nnz);
}

} // namespace

int main(int argc, char **argv) {
    try {
        cellerator::bench::benchmark_mutex_guard benchmark_mutex("cellShardRealSpmmBench");
        const bench_config cfg = parse_args(argc, argv);
        std::vector<real_case> cases = load_manifest(cfg.manifest_path, cfg.dataset_filter);
        autograd::fleet_context fleet;
        autograd::init(&fleet);
        autograd::discover_fleet(&fleet, true, cudaStreamNonBlocking, true);
        require(fleet.local.device_count >= 2u, "cellShardRealSpmmBench requires at least 2 visible GPUs");
        for (const real_case &entry : cases) run_case(entry, cfg, &fleet);
        autograd::clear(&fleet);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "cellShardRealSpmmBench: %s\n", e.what());
        return 1;
    }
}
