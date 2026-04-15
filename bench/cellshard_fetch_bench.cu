#include "../extern/CellShard/src/CellShard.hh"
#include "benchmark_mutex.hh"
#include "../src/ingest/mtx/mtx_reader.cuh"

#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

namespace cs = ::cellshard;
namespace cmtx = ::cellerator::ingest::mtx;

namespace {

struct bench_config {
    unsigned long parts = 64ul;
    unsigned long rows_per_partition = 1024ul;
    unsigned long cols = 4096ul;
    unsigned long shards = 8ul;
    unsigned int block_size = 16u;
    unsigned int ell_cols = 64u;
    unsigned int warmup = 10u;
    unsigned int iters = 50u;
    unsigned long shard_id = std::numeric_limits<unsigned long>::max();
    unsigned long reader_bytes_mb = 8ul;
    std::string artifact_root = "/tmp";
    std::string artifact_dir;
    std::string impl = "all";
    bool use_real_data = false;
    bool include_synthetic = true;
    bool prepare_only = false;
    bool keep_files = false;
    std::vector<std::string> real_mtx_paths;
};

struct run_result {
    double cold_ms = 0.0;
    double warm_avg_ms = 0.0;
};

struct temp_paths {
    std::string base_dir;
    std::string csh5_path;
    std::string cache_dir;
};

struct scenario_case {
    std::string label;
    std::string source_kind;
    std::string matrix_path;
};

struct file_size_result {
    std::uint64_t bytes = 0u;
};

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

unsigned long parse_ul(const char *value, const char *label) {
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (value == nullptr || *value == '\0' || end == nullptr || *end != '\0') {
        throw std::invalid_argument(std::string("invalid integer for ") + label);
    }
    return parsed;
}

bench_config parse_args(int argc, char **argv) {
    bench_config cfg;

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        auto require_value = [&](const char *label) -> const char * {
            if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + label);
            return argv[++i];
        };
        if (arg == "--parts") {
            cfg.parts = parse_ul(require_value("--parts"), "--parts");
        } else if (arg == "--rows-per-partition") {
            cfg.rows_per_partition = parse_ul(require_value("--rows-per-partition"), "--rows-per-partition");
        } else if (arg == "--cols") {
            cfg.cols = parse_ul(require_value("--cols"), "--cols");
        } else if (arg == "--shards") {
            cfg.shards = parse_ul(require_value("--shards"), "--shards");
        } else if (arg == "--block-size") {
            cfg.block_size = (unsigned int) parse_ul(require_value("--block-size"), "--block-size");
        } else if (arg == "--ell-cols") {
            cfg.ell_cols = (unsigned int) parse_ul(require_value("--ell-cols"), "--ell-cols");
        } else if (arg == "--warmup") {
            cfg.warmup = (unsigned int) parse_ul(require_value("--warmup"), "--warmup");
        } else if (arg == "--iters") {
            cfg.iters = (unsigned int) parse_ul(require_value("--iters"), "--iters");
        } else if (arg == "--shard-id") {
            cfg.shard_id = parse_ul(require_value("--shard-id"), "--shard-id");
        } else if (arg == "--reader-bytes-mb") {
            cfg.reader_bytes_mb = parse_ul(require_value("--reader-bytes-mb"), "--reader-bytes-mb");
        } else if (arg == "--artifact-root") {
            cfg.artifact_root = require_value("--artifact-root");
        } else if (arg == "--artifact-dir") {
            cfg.artifact_dir = require_value("--artifact-dir");
        } else if (arg == "--impl") {
            cfg.impl = require_value("--impl");
        } else if (arg == "--real-data") {
            cfg.use_real_data = true;
        } else if (arg == "--real-mtx") {
            cfg.real_mtx_paths.emplace_back(require_value("--real-mtx"));
        } else if (arg == "--skip-synthetic") {
            cfg.include_synthetic = false;
        } else if (arg == "--prepare-only") {
            cfg.prepare_only = true;
            cfg.keep_files = true;
        } else if (arg == "--keep-files") {
            cfg.keep_files = true;
        } else if (arg == "-h" || arg == "--help") {
            std::cout
                << "Usage: cellShardFetchBench [--parts N] [--rows-per-partition N] [--cols N] [--shards N]\n"
                << "                            [--block-size N] [--ell-cols N] [--warmup N] [--iters N]\n"
                << "                            [--shard-id N] [--artifact-root PATH] [--artifact-dir PATH]\n"
                << "                            [--reader-bytes-mb N] [--impl NAME] [--real-data] [--real-mtx PATH]\n"
                << "                            [--skip-synthetic] [--prepare-only] [--keep-files]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument(std::string("unknown argument: ") + arg);
        }
    }

    if (cfg.parts == 0ul) throw std::invalid_argument("--parts must be > 0");
    if (cfg.rows_per_partition == 0ul) throw std::invalid_argument("--rows-per-partition must be > 0");
    if (cfg.cols == 0ul) throw std::invalid_argument("--cols must be > 0");
    if (cfg.shards == 0ul) throw std::invalid_argument("--shards must be > 0");
    if (cfg.block_size == 0u) throw std::invalid_argument("--block-size must be > 0");
    if (cfg.ell_cols == 0u || (cfg.ell_cols % cfg.block_size) != 0u) {
        throw std::invalid_argument("--ell-cols must be a non-zero multiple of --block-size");
    }
    if (cfg.impl != "all" && cfg.impl != "csh5" && cfg.impl != "csh5_cache") {
        throw std::invalid_argument("--impl must be one of: all, csh5, csh5_cache");
    }
    if (!cfg.artifact_dir.empty() && cfg.prepare_only) {
        throw std::invalid_argument("--artifact-dir cannot be combined with --prepare-only");
    }
    return cfg;
}

std::string sanitize_label(const std::string &label) {
    std::string out;
    out.reserve(label.size());
    for (char c : label) {
        if ((c >= 'a' && c <= 'z')
            || (c >= 'A' && c <= 'Z')
            || (c >= '0' && c <= '9')
            || c == '-' || c == '_') {
            out.push_back(c);
        } else {
            out.push_back('_');
        }
    }
    return out.empty() ? std::string("scenario") : out;
}

temp_paths make_temp_paths(const std::string &artifact_root, const std::string &label) {
    std::string templ_str = artifact_root + "/cellshard_fetch_bench." + sanitize_label(label) + ".XXXXXX";
    std::vector<char> templ(templ_str.begin(), templ_str.end());
    templ.push_back('\0');
    char *dir = ::mkdtemp(templ.data());
    if (dir == nullptr) throw std::runtime_error("mkdtemp failed");

    temp_paths out;
    out.base_dir = dir;
    out.csh5_path = out.base_dir + "/dataset.csh5";
    out.cache_dir = out.base_dir + "/cache";
    return out;
}

temp_paths open_existing_paths(const std::string &artifact_dir) {
    temp_paths out;
    out.base_dir = artifact_dir;
    out.csh5_path = out.base_dir + "/dataset.csh5";
    out.cache_dir = out.base_dir + "/cache";
    return out;
}

void cleanup_paths(const temp_paths &paths) {
    std::error_code ec;
    std::filesystem::remove_all(paths.cache_dir, ec);
    ec.clear();
    std::filesystem::remove(paths.csh5_path, ec);
    ec.clear();
    std::filesystem::remove_all(paths.base_dir, ec);
}

file_size_result stat_file(const std::string &path) {
    struct stat st;
    file_size_result out;
    if (::stat(path.c_str(), &st) != 0) throw std::runtime_error("stat failed");
    out.bytes = (std::uint64_t) st.st_size;
    return out;
}

double elapsed_ms(const std::chrono::steady_clock::time_point &start,
                  const std::chrono::steady_clock::time_point &stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

void fill_blocked_ell_part(cs::sparse::blocked_ell *part, unsigned long seed) {
    const std::size_t row_blocks = (std::size_t) cs::sparse::row_block_count(part);
    const std::size_t ell_width = (std::size_t) cs::sparse::ell_width_blocks(part);
    const std::size_t value_count = (std::size_t) part->rows * (std::size_t) part->ell_cols;

    for (std::size_t rb = 0; rb < row_blocks; ++rb) {
        for (std::size_t slot = 0; slot < ell_width; ++slot) {
            const std::size_t idx = rb * ell_width + slot;
            part->blockColIdx[idx] = (cs::types::idx_t) ((rb + slot + seed) % ell_width);
        }
    }
    for (std::size_t i = 0; i < value_count; ++i) {
        part->val[i] = __float2half(0.125f + (float) ((i + seed) % 29u) * 0.03125f);
    }
}

void build_source_matrix(const bench_config &cfg, cs::sharded<cs::sparse::blocked_ell> *out) {
    cs::init(out);
    for (unsigned long part_id = 0; part_id < cfg.parts; ++part_id) {
        cs::sparse::blocked_ell *part = new cs::sparse::blocked_ell;
        cs::sparse::init(part,
                         (cs::types::dim_t) cfg.rows_per_partition,
                         (cs::types::dim_t) cfg.cols,
                         (cs::types::nnz_t) ((std::size_t) cfg.rows_per_partition * (std::size_t) cfg.ell_cols),
                         (cs::types::u32) cfg.block_size,
                         (cs::types::u32) cfg.ell_cols);
        if (!cs::sparse::allocate(part)) {
            cs::sparse::clear(part);
            delete part;
            throw std::runtime_error("failed to allocate blocked ell part");
        }
        fill_blocked_ell_part(part, part_id);
        if (!cs::append_partition(out, part)) {
            cs::sparse::clear(part);
            delete part;
            throw std::runtime_error("append_partition failed");
        }
    }
    if (cfg.shards >= cfg.parts) {
        if (!cs::set_shards_to_partitions(out)) throw std::runtime_error("set_shards_to_partitions failed");
    } else {
        if (!cs::set_equal_shards(out, cfg.shards)) throw std::runtime_error("set_equal_shards failed");
    }
}

std::vector<unsigned long> build_real_row_offsets(unsigned long rows, unsigned long rows_per_partition) {
    std::vector<unsigned long> offsets;
    unsigned long row = 0;

    if (rows_per_partition == 0ul) throw std::invalid_argument("rows_per_partition must be > 0");
    offsets.push_back(0ul);
    while (row < rows) {
        row = std::min(rows, row + rows_per_partition);
        offsets.push_back(row);
    }
    return offsets;
}

void convert_coo_part_to_csr_cpu(const cs::sparse::coo *src, cs::sparse::compressed *dst) {
    std::vector<unsigned int> next;

    if (src == nullptr || dst == nullptr) throw std::invalid_argument("null coo/csr pointer");
    cs::sparse::init(dst,
                     (cs::types::dim_t) src->rows,
                     (cs::types::dim_t) src->cols,
                     (cs::types::nnz_t) src->nnz,
                     cs::sparse::compressed_by_row);
    if (!cs::sparse::allocate(dst)) {
        throw std::runtime_error("failed to allocate csr part");
    }
    std::fill_n(dst->majorPtr, (std::size_t) src->rows + 1u, 0u);
    for (cs::types::nnz_t i = 0; i < src->nnz; ++i) {
        ++dst->majorPtr[src->rowIdx[i] + 1u];
    }
    for (cs::types::dim_t row = 0; row < src->rows; ++row) {
        dst->majorPtr[row + 1u] += dst->majorPtr[row];
    }
    next.assign(dst->majorPtr, dst->majorPtr + src->rows);
    for (cs::types::nnz_t i = 0; i < src->nnz; ++i) {
        const unsigned int row = src->rowIdx[i];
        const unsigned int pos = next[row]++;
        dst->minorIdx[pos] = src->colIdx[i];
        dst->val[pos] = src->val[i];
    }
}

std::string default_real_label_from_path(const std::string &path) {
    const std::size_t exon_pos = path.find("/exon/matrix.mtx");
    const std::size_t slash = path.find_last_of('/', exon_pos == std::string::npos ? path.size() : exon_pos);
    if (exon_pos != std::string::npos && slash != std::string::npos && slash > 0u) {
        const std::size_t prev = path.find_last_of('/', slash - 1u);
        if (prev != std::string::npos && prev + 1u < slash) {
            return path.substr(prev + 1u, exon_pos - prev - 1u) + "_exon";
        }
    }
    const std::size_t name_begin = path.find_last_of('/');
    return name_begin == std::string::npos ? path : path.substr(name_begin + 1u);
}

void build_real_source_matrix(const bench_config &cfg,
                              const std::string &matrix_path,
                              cs::sharded<cs::sparse::blocked_ell> *out) {
    static constexpr unsigned int kBlockedEllCandidates[] = {8u, 16u, 32u};
    cmtx::header header;
    cs::sharded<cs::sparse::coo> coo;
    unsigned long *row_nnz = nullptr;
    unsigned long *part_nnz = nullptr;
    const std::size_t reader_bytes = (std::size_t) cfg.reader_bytes_mb << 20u;
    std::vector<unsigned long> row_offsets;
    int ok = 0;

    cmtx::init(&header);
    cs::init(out);
    cs::init(&coo);

    try {
        if (!cmtx::scan_row_nnz(matrix_path.c_str(), &header, &row_nnz, reader_bytes)) {
            throw std::runtime_error("scan_row_nnz failed for real data");
        }

        row_offsets = build_real_row_offsets(header.rows, cfg.rows_per_partition);
        if (!cmtx::count_all_part_nnz(matrix_path.c_str(),
                                      &header,
                                      row_offsets.data(),
                                      (unsigned long) row_offsets.size() - 1ul,
                                      &part_nnz)) {
            throw std::runtime_error("count_all_part_nnz failed for real data");
        }
        if (!cmtx::load_part_window_coo(matrix_path.c_str(),
                                        &header,
                                        row_offsets.data(),
                                        part_nnz,
                                        (unsigned long) row_offsets.size() - 1ul,
                                        0ul,
                                        (unsigned long) row_offsets.size() - 1ul,
                                        &coo,
                                        reader_bytes)) {
            throw std::runtime_error("load_part_window_coo failed for real data");
        }

        for (unsigned long part_id = 0; part_id < coo.num_partitions; ++part_id) {
            cs::sparse::compressed compressed;
            cs::convert::blocked_ell_tune_result tune{};
            cs::sparse::blocked_ell *blocked = new cs::sparse::blocked_ell;
            cs::sparse::init(&compressed);
            cs::sparse::init(blocked);
            try {
                convert_coo_part_to_csr_cpu(coo.parts[part_id], &compressed);
            } catch (...) {
                cs::sparse::clear(&compressed);
                cs::sparse::clear(blocked);
                delete blocked;
                throw;
            }
            ok = cs::convert::blocked_ell_from_compressed_auto(&compressed,
                                                               kBlockedEllCandidates,
                                                               sizeof(kBlockedEllCandidates) / sizeof(kBlockedEllCandidates[0]),
                                                               blocked,
                                                               &tune);
            cs::sparse::clear(&compressed);
            if (!ok) {
                cs::sparse::clear(blocked);
                delete blocked;
                throw std::runtime_error("blocked_ell_from_compressed_auto failed for real data");
            }
            if (!cs::append_partition(out, blocked)) {
                cs::sparse::clear(blocked);
                delete blocked;
                throw std::runtime_error("append_partition failed for real data");
            }
        }

        if (cfg.shards >= out->num_partitions) {
            if (!cs::set_shards_to_partitions(out)) throw std::runtime_error("set_shards_to_partitions failed for real data");
        } else {
            if (!cs::set_equal_shards(out, cfg.shards)) throw std::runtime_error("set_equal_shards failed for real data");
        }
    } catch (...) {
        cs::clear(out);
        cs::clear(&coo);
        std::free(part_nnz);
        std::free(row_nnz);
        throw;
    }

    cs::clear(&coo);
    std::free(part_nnz);
    std::free(row_nnz);
}

void write_csh5_from_source(const std::string &path, const cs::sharded<cs::sparse::blocked_ell> &source) {
    std::vector<std::uint64_t> part_rows(source.num_partitions, 0u);
    std::vector<std::uint64_t> part_nnz(source.num_partitions, 0u);
    std::vector<std::uint64_t> part_aux(source.num_partitions, 0u);
    std::vector<std::uint64_t> part_row_offsets(source.num_partitions + 1u, 0u);
    std::vector<std::uint32_t> part_dataset_ids(source.num_partitions, 0u);
    std::vector<std::uint32_t> part_codec_ids(source.num_partitions, 0u);
    std::vector<std::uint64_t> shard_offsets(source.num_shards + 1u, 0u);
    cs::dataset_codec_descriptor codec{};
    cs::dataset_layout_view layout{};

    for (unsigned long i = 0; i < source.num_partitions; ++i) {
        part_rows[i] = (std::uint64_t) source.partition_rows[i];
        part_nnz[i] = (std::uint64_t) source.partition_nnz[i];
        part_aux[i] = (std::uint64_t) source.partition_aux[i];
        part_row_offsets[i] = (std::uint64_t) source.partition_offsets[i];
    }
    part_row_offsets[source.num_partitions] = (std::uint64_t) source.partition_offsets[source.num_partitions];
    for (unsigned long i = 0; i <= source.num_shards; ++i) {
        shard_offsets[i] = (std::uint64_t) source.shard_offsets[i];
    }

    codec.codec_id = 0u;
    codec.family = cs::dataset_codec_family_blocked_ell;
    codec.value_code = (std::uint32_t) ::real::code_of< ::real::storage_t>::code;
    codec.scale_value_code = 0u;
    codec.bits = (std::uint32_t) (sizeof(::real::storage_t) * 8u);
    codec.flags = 0u;

    layout.rows = source.rows;
    layout.cols = source.cols;
    layout.nnz = source.nnz;
    layout.num_partitions = source.num_partitions;
    layout.num_shards = source.num_shards;
    layout.partition_rows = part_rows.data();
    layout.partition_nnz = part_nnz.data();
    layout.partition_axes = 0;
    layout.partition_aux = part_aux.data();
    layout.partition_row_offsets = part_row_offsets.data();
    layout.partition_dataset_ids = part_dataset_ids.data();
    layout.partition_codec_ids = part_codec_ids.data();
    layout.shard_offsets = shard_offsets.data();
    layout.codecs = &codec;
    layout.num_codecs = 1u;

    if (!cs::create_dataset_blocked_ell_h5(path.c_str(), &layout, 0, 0)) {
        throw std::runtime_error("create_dataset_blocked_ell_h5 failed");
    }
    for (unsigned long i = 0; i < source.num_partitions; ++i) {
        if (!cs::append_blocked_ell_partition_h5(path.c_str(), i, source.parts[i])) {
            throw std::runtime_error("append_blocked_ell_partition_h5 failed");
        }
    }
}

bool equal_blocked_ell_part(const cs::sparse::blocked_ell *a, const cs::sparse::blocked_ell *b) {
    const std::size_t idx_count = (std::size_t) cs::sparse::row_block_count(a) * (std::size_t) cs::sparse::ell_width_blocks(a);
    const std::size_t value_count = (std::size_t) a->rows * (std::size_t) a->ell_cols;

    if (a == nullptr || b == nullptr) return false;
    if (a->rows != b->rows || a->cols != b->cols || a->nnz != b->nnz || a->block_size != b->block_size || a->ell_cols != b->ell_cols) return false;
    for (std::size_t i = 0; i < idx_count; ++i) {
        if (a->blockColIdx[i] != b->blockColIdx[i]) return false;
    }
    for (std::size_t i = 0; i < value_count; ++i) {
        if (__half2float(a->val[i]) != __half2float(b->val[i])) return false;
    }
    return true;
}

void validate_loaded_shard(const cs::sharded<cs::sparse::blocked_ell> &reference,
                           const cs::sharded<cs::sparse::blocked_ell> &loaded,
                           unsigned long shard_id) {
    const unsigned long begin = cs::first_partition_in_shard(&reference, shard_id);
    const unsigned long end = cs::last_partition_in_shard(&reference, shard_id);
    for (unsigned long part_id = begin; part_id < end; ++part_id) {
        if (!equal_blocked_ell_part(reference.parts[part_id], loaded.parts[part_id])) {
            throw std::runtime_error("loaded shard did not match reference payload");
        }
    }
}

void prefetch_csh5_cache(const std::string &path, const std::string &cache_dir, unsigned long shard_id) {
    cs::sharded<cs::sparse::blocked_ell> loaded;
    cs::shard_storage storage;

    cs::init(&loaded);
    cs::init(&storage);
    if (!cs::load_header(path.c_str(), &loaded, &storage)) {
        throw std::runtime_error("csh5 load_header failed during prefetch");
    }
    if (!cs::bind_dataset_h5_cache(&storage, cache_dir.c_str())) {
        throw std::runtime_error("bind_dataset_h5_cache failed during prefetch");
    }
    if (!cs::prefetch_dataset_blocked_ell_h5_shard_cache(&loaded, &storage, shard_id)) {
        throw std::runtime_error("prefetch_dataset_blocked_ell_h5_shard_cache failed");
    }
    cs::clear(&storage);
    cs::clear(&loaded);
}

run_result benchmark_csh5(const std::string &path,
                          const std::string &cache_dir,
                          bool use_cache,
                          const cs::sharded<cs::sparse::blocked_ell> &reference,
                          unsigned long shard_id,
                          unsigned int warmup,
                          unsigned int iters) {
    cs::sharded<cs::sparse::blocked_ell> loaded;
    cs::shard_storage storage;
    run_result out{};

    cs::init(&loaded);
    cs::init(&storage);
    if (!cs::load_header(path.c_str(), &loaded, &storage)) {
        throw std::runtime_error("csh5 load_header failed");
    }
    if (use_cache) {
        if (!cs::bind_dataset_h5_cache(&storage, cache_dir.c_str())) {
            throw std::runtime_error("bind_dataset_h5_cache failed");
        }
    }

    {
        const auto start = std::chrono::steady_clock::now();
        if (!cs::fetch_shard(&loaded, &storage, shard_id)) throw std::runtime_error("csh5 cold fetch_shard failed");
        const auto stop = std::chrono::steady_clock::now();
        out.cold_ms = elapsed_ms(start, stop);
    }
    validate_loaded_shard(reference, loaded, shard_id);
    if (!cs::drop_shard(&loaded, shard_id)) throw std::runtime_error("csh5 drop_shard failed");

    for (unsigned int i = 0; i < warmup; ++i) {
        if (!cs::fetch_shard(&loaded, &storage, shard_id)) throw std::runtime_error("csh5 warmup fetch_shard failed");
        if (!cs::drop_shard(&loaded, shard_id)) throw std::runtime_error("csh5 warmup drop_shard failed");
    }
    {
        const auto start = std::chrono::steady_clock::now();
        for (unsigned int i = 0; i < iters; ++i) {
            if (!cs::fetch_shard(&loaded, &storage, shard_id)) throw std::runtime_error("csh5 warm fetch_shard failed");
            if (!cs::drop_shard(&loaded, shard_id)) throw std::runtime_error("csh5 warm drop_shard failed");
        }
        const auto stop = std::chrono::steady_clock::now();
        out.warm_avg_ms = elapsed_ms(start, stop) / (double) iters;
    }

    cs::clear(&storage);
    cs::clear(&loaded);
    return out;
}

void load_reference_from_csh5(const std::string &path,
                              const std::string &cache_dir,
                              cs::sharded<cs::sparse::blocked_ell> *out) {
    cs::shard_storage storage;
    cs::init(out);
    cs::init(&storage);
    if (!cs::load_header(path.c_str(), out, &storage)) {
        throw std::runtime_error("failed to load csh5 header for artifact reuse");
    }
    if (!cs::bind_dataset_h5_cache(&storage, cache_dir.c_str())) {
        cs::clear(&storage);
        throw std::runtime_error("failed to bind csh5 cache for artifact reuse");
    }
    if (!cs::fetch_all_partitions(out, &storage)) {
        cs::clear(&storage);
        throw std::runtime_error("failed to materialize csh5 reference for artifact reuse");
    }
    cs::clear(&storage);
}

std::uint64_t shard_payload_bytes(const cs::sharded<cs::sparse::blocked_ell> &source, unsigned long shard_id) {
    const unsigned long begin = cs::first_partition_in_shard(&source, shard_id);
    const unsigned long end = cs::last_partition_in_shard(&source, shard_id);
    std::uint64_t bytes = 0u;
    for (unsigned long i = begin; i < end; ++i) {
        bytes += (std::uint64_t) cs::partition_bytes(&source, i);
    }
    return bytes;
}

void print_result(const char *impl_name,
                  const run_result &result,
                  std::uint64_t file_bytes,
                  std::uint64_t shard_bytes,
                  double baseline_warm_ms,
                  bool has_baseline) {
    const double warm_delta_pct = has_baseline && baseline_warm_ms > 0.0
        ? ((result.warm_avg_ms - baseline_warm_ms) / baseline_warm_ms) * 100.0
        : 0.0;
    const double shard_mb = (double) shard_bytes / (1024.0 * 1024.0);
    const double warm_gib_per_s = result.warm_avg_ms > 0.0
        ? ((double) shard_bytes / (1024.0 * 1024.0 * 1024.0)) / (result.warm_avg_ms * 1.0e-3)
        : 0.0;

    std::cout
        << "impl=" << impl_name
        << " file_bytes=" << file_bytes
        << " shard_payload_mb=" << shard_mb
        << " cold_ms=" << result.cold_ms
        << " warm_avg_ms=" << result.warm_avg_ms
        << " warm_gib_per_s=" << warm_gib_per_s
        << " warm_delta_vs_baseline_pct=";
    if (has_baseline) std::cout << warm_delta_pct;
    else std::cout << "na";
    std::cout << '\n';
}

void run_case(const bench_config &cfg, const scenario_case &scenario) {
    const bool reuse_artifacts = !cfg.artifact_dir.empty();
    const bool run_csh5 = (cfg.impl == "all" || cfg.impl == "csh5");
    const bool run_csh5_cache = (cfg.impl == "all" || cfg.impl == "csh5_cache");
    const bool owns_paths = !reuse_artifacts;
    temp_paths paths = reuse_artifacts ? open_existing_paths(cfg.artifact_dir) : make_temp_paths(cfg.artifact_root, scenario.label);
    cs::sharded<cs::sparse::blocked_ell> source;
    unsigned long shard_id = cfg.shard_id;

    cs::init(&source);

    try {
        if (reuse_artifacts) {
            load_reference_from_csh5(paths.csh5_path, paths.cache_dir, &source);
        } else {
            if (scenario.matrix_path.empty()) {
                build_source_matrix(cfg, &source);
            } else {
                build_real_source_matrix(cfg, scenario.matrix_path, &source);
            }
            write_csh5_from_source(paths.csh5_path, source);
        }
        if (source.num_shards == 0ul) throw std::runtime_error("source.num_shards must be > 0");
        if (shard_id == std::numeric_limits<unsigned long>::max()) {
            shard_id = source.num_shards > 1ul ? source.num_shards / 2ul : 0ul;
        }
        require(shard_id < source.num_shards, "--shard-id must be < num_shards");

        const file_size_result csh5_size = stat_file(paths.csh5_path);
        const std::uint64_t selected_shard_bytes = shard_payload_bytes(source, shard_id);
        const unsigned long shard_begin = cs::first_partition_in_shard(&source, shard_id);
        const unsigned long shard_end = cs::last_partition_in_shard(&source, shard_id);
        run_result csh5{};
        run_result csh5_cache{};

        if (cfg.prepare_only) {
            std::cout
                << "prepared_artifact_dir=" << paths.base_dir
                << " scenario=" << scenario.label
                << " source=" << scenario.source_kind
                << " target_shard=" << shard_id
                << " parts=" << source.num_partitions
                << " shards=" << source.num_shards
                << '\n';
            return;
        }

        if (run_csh5_cache) prefetch_csh5_cache(paths.csh5_path, paths.cache_dir, shard_id);
        if (run_csh5) csh5 = benchmark_csh5(paths.csh5_path, paths.cache_dir, false, source, shard_id, cfg.warmup, cfg.iters);
        if (run_csh5_cache) csh5_cache = benchmark_csh5(paths.csh5_path, paths.cache_dir, true, source, shard_id, cfg.warmup, cfg.iters);

        std::cout
            << "scenario=" << scenario.label
            << " source=" << scenario.source_kind
            << " storage=ssd"
            << " artifact_root=" << cfg.artifact_root
            << " format=blocked_ell"
            << " rows=" << source.rows
            << " cols=" << source.cols
            << " parts=" << source.num_partitions
            << " shards=" << source.num_shards
            << " target_shard=" << shard_id
            << " parts_in_shard=" << (shard_end - shard_begin)
            << " warmup=" << cfg.warmup
            << " iters=" << cfg.iters;
        if (!scenario.matrix_path.empty()) {
            std::cout << " matrix_path=" << scenario.matrix_path;
        } else {
            std::cout
                << " rows_per_partition=" << cfg.rows_per_partition
                << " block_size=" << cfg.block_size
                << " ell_cols=" << cfg.ell_cols;
        }
        std::cout << '\n';

        if (run_csh5) print_result("csh5", csh5, csh5_size.bytes, selected_shard_bytes, 0.0, false);
        if (run_csh5_cache) print_result("csh5_cache",
                                         csh5_cache,
                                         csh5_size.bytes,
                                         selected_shard_bytes,
                                         csh5.warm_avg_ms,
                                         run_csh5);
    } catch (...) {
        cs::clear(&source);
        if (owns_paths && !paths.base_dir.empty() && !cfg.keep_files) cleanup_paths(paths);
        throw;
    }

    {
        cs::clear(&source);
        if (owns_paths && !paths.base_dir.empty() && !cfg.keep_files) cleanup_paths(paths);
    }
}

} // namespace

int main(int argc, char **argv) {
    static const char *kDefaultEmbryoExon1 = "/home/tumlinson/embryo_scratch/embryo_1/exon/matrix.mtx";
    static const char *kDefaultEmbryoExon15 = "/home/tumlinson/embryo_scratch/embryo_15/exon/matrix.mtx";
    int rc = 1;
    bench_config cfg;
    std::vector<scenario_case> scenarios;

    try {
        cfg = parse_args(argc, argv);
        cellerator::bench::benchmark_mutex_guard guard("cellshard-fetch");
        if (cfg.include_synthetic) scenarios.push_back({"synthetic_blocked_ell", "synthetic", ""});
        if (cfg.use_real_data && cfg.real_mtx_paths.empty()) {
            cfg.real_mtx_paths.push_back(kDefaultEmbryoExon1);
            cfg.real_mtx_paths.push_back(kDefaultEmbryoExon15);
        }
        for (const std::string &path : cfg.real_mtx_paths) {
            scenarios.push_back({default_real_label_from_path(path), "real_mtx", path});
        }
        if (scenarios.empty()) {
            throw std::runtime_error("no scenarios selected");
        }
        for (const scenario_case &scenario : scenarios) {
            run_case(cfg, scenario);
        }
        rc = 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "cellShardFetchBench failed: %s\n", e.what());
    }

    return rc;
}
