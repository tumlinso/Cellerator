#include "benchmark_mutex.hh"
#include <Cellerator/workbench/dataset_workbench.hh>
#include "../extern/CellShard/include/CellShard/CellShard.hh"

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <hdf5.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace wb = ::cellerator::apps::workbench;
namespace cs = ::cellshard;
namespace csc = ::cellshard::convert;

static constexpr unsigned char analysis_magic[8] = { 'C', 'P', 'R', 'A', '1', 0, 0, 0 };

struct config {
    std::string h5ad_path;
    std::string output_dir;
    std::string cache_root;
    std::string working_root;
    std::string matrix_source = "raw_x";
    std::string python_exe = "python3";
    std::size_t reader_bytes = (std::size_t) 8u << 20u;
    unsigned long max_part_nnz = 1ul << 24ul;
    unsigned long convert_window_bytes = 1ul << 28ul;
    unsigned long target_shard_bytes = 1ul << 29ul;
    unsigned int warmup = 1u;
    unsigned int repeats = 1u;
    int device = 0;
    float target_sum = 10000.0f;
    float min_counts = 500.0f;
    unsigned int min_genes = 200u;
    float max_mito_fraction = 0.2f;
    float min_gene_sum = 1.0f;
    float min_detected_cells = 5.0f;
    float min_variance = 0.01f;
    std::uint32_t sliced_bucket_count = 4u;
    bool use_all_devices = true;
    bool run_blocked_analysis = false;
    bool run_finalize = false;
    bool run_python_reference = false;
    bool reuse_artifacts = true;
};

struct scoped_nvtx_range {
    explicit scoped_nvtx_range(const char *label) { nvtxRangePushA(label); }
    ~scoped_nvtx_range() { nvtxRangePop(); }
};

struct dataset_artifacts {
    fs::path blocked_input;
    fs::path sliced_input;
    fs::path blocked_final;
    fs::path blocked_analysis_blob;
    fs::path sliced_analysis_blob;
    fs::path stability_summary_tsv;
    fs::path stability_details_json;
    fs::path run_config_json;
    fs::path results_json;
    fs::path summary_json;
    fs::path summary_txt;
};

static void usage(const char *argv0) {
    std::fprintf(stderr,
                 "Usage: %s --h5ad PATH --output-dir PATH [options]\n"
                 "  --h5ad PATH                 Real H5AD input. Required.\n"
                 "  --output-dir PATH           Benchmark artifact directory. Required.\n"
                 "  --cache-root PATH           Dataset cache root. Default: OUTPUT/cache.\n"
                 "  --working-root PATH         Working root for finalize/repack. Default: OUTPUT/work.\n"
                 "  --matrix-source NAME        H5AD matrix source. Default: raw_x.\n"
                 "  --python-exe PATH           Python interpreter for baseline. Default: python3.\n"
                 "  --reader-bytes-mb N         H5AD reader window in MiB. Default: 8.\n"
                 "  --max-part-nnz N            Max nnz per physical partition. Default: 16777216.\n"
                 "  --convert-window-mb N       COO load/convert window in MiB. Default: 256.\n"
                 "  --target-shard-mb N         Target optimized shard bytes in MiB. Default: 512.\n"
                 "  --warmup N                  Warmup passes. Default: 1.\n"
                 "  --repeats N                 Measured passes. Default: 1.\n"
                 "  --device N                  CUDA device. Default: 0.\n"
                 "  --target-sum F              Normalize target. Default: 10000.\n"
                 "  --min-counts F              Cell min counts. Default: 500.\n"
                 "  --min-genes N               Cell min genes. Default: 200.\n"
                 "  --max-mito-fraction F       Cell max mito fraction. Default: 0.2.\n"
                 "  --min-gene-sum F            Gene min sum. Default: 1.\n"
                 "  --min-detected-cells F      Gene min detected cells. Default: 5.\n"
                 "  --min-variance F            Gene min variance. Default: 0.01.\n"
                 "  --sliced-bucket-count N     Sliced execution bucket target. Default: 4.\n"
                 "  --single-device             Disable multi-GPU preprocess and use --device only.\n"
                 "  --with-blocked-analysis     Run blocked preprocess compatibility pass.\n"
                 "  --with-finalize             Run blocked finalize+browse stage.\n"
                 "  --with-python-reference     Run Python numerical-stability baseline.\n"
                 "  --no-reuse-artifacts        Rebuild input artifacts even if present.\n",
                 argv0);
}

static int parse_u32(const char *text, unsigned int *value) {
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (text == end || *end != 0 || parsed > 0xfffffffful) return 0;
    *value = (unsigned int) parsed;
    return 1;
}

static int parse_i32(const char *text, int *value) {
    char *end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (text == end || *end != 0 || parsed < (long) std::numeric_limits<int>::min()
        || parsed > (long) std::numeric_limits<int>::max()) {
        return 0;
    }
    *value = (int) parsed;
    return 1;
}

static int parse_f32(const char *text, float *value) {
    char *end = nullptr;
    const float parsed = std::strtof(text, &end);
    if (text == end || *end != 0) return 0;
    *value = parsed;
    return 1;
}

static int parse_args(int argc, char **argv, config *cfg) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--h5ad") == 0 && i + 1 < argc) {
            cfg->h5ad_path = argv[++i];
        } else if (std::strcmp(argv[i], "--output-dir") == 0 && i + 1 < argc) {
            cfg->output_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--cache-root") == 0 && i + 1 < argc) {
            cfg->cache_root = argv[++i];
        } else if (std::strcmp(argv[i], "--working-root") == 0 && i + 1 < argc) {
            cfg->working_root = argv[++i];
        } else if (std::strcmp(argv[i], "--matrix-source") == 0 && i + 1 < argc) {
            cfg->matrix_source = argv[++i];
        } else if (std::strcmp(argv[i], "--python-exe") == 0 && i + 1 < argc) {
            cfg->python_exe = argv[++i];
        } else if (std::strcmp(argv[i], "--reader-bytes-mb") == 0 && i + 1 < argc) {
            unsigned int mb = 0u;
            if (!parse_u32(argv[++i], &mb)) return 0;
            cfg->reader_bytes = (std::size_t) mb << 20u;
        } else if (std::strcmp(argv[i], "--max-part-nnz") == 0 && i + 1 < argc) {
            unsigned int value = 0u;
            if (!parse_u32(argv[++i], &value)) return 0;
            cfg->max_part_nnz = (unsigned long) value;
        } else if (std::strcmp(argv[i], "--convert-window-mb") == 0 && i + 1 < argc) {
            unsigned int mb = 0u;
            if (!parse_u32(argv[++i], &mb)) return 0;
            cfg->convert_window_bytes = (unsigned long) mb << 20u;
        } else if (std::strcmp(argv[i], "--target-shard-mb") == 0 && i + 1 < argc) {
            unsigned int mb = 0u;
            if (!parse_u32(argv[++i], &mb)) return 0;
            cfg->target_shard_bytes = (unsigned long) mb << 20u;
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->warmup)) return 0;
        } else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->repeats)) return 0;
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            if (!parse_i32(argv[++i], &cfg->device)) return 0;
        } else if (std::strcmp(argv[i], "--target-sum") == 0 && i + 1 < argc) {
            if (!parse_f32(argv[++i], &cfg->target_sum)) return 0;
        } else if (std::strcmp(argv[i], "--min-counts") == 0 && i + 1 < argc) {
            if (!parse_f32(argv[++i], &cfg->min_counts)) return 0;
        } else if (std::strcmp(argv[i], "--min-genes") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->min_genes)) return 0;
        } else if (std::strcmp(argv[i], "--max-mito-fraction") == 0 && i + 1 < argc) {
            if (!parse_f32(argv[++i], &cfg->max_mito_fraction)) return 0;
        } else if (std::strcmp(argv[i], "--min-gene-sum") == 0 && i + 1 < argc) {
            if (!parse_f32(argv[++i], &cfg->min_gene_sum)) return 0;
        } else if (std::strcmp(argv[i], "--min-detected-cells") == 0 && i + 1 < argc) {
            if (!parse_f32(argv[++i], &cfg->min_detected_cells)) return 0;
        } else if (std::strcmp(argv[i], "--min-variance") == 0 && i + 1 < argc) {
            if (!parse_f32(argv[++i], &cfg->min_variance)) return 0;
        } else if (std::strcmp(argv[i], "--sliced-bucket-count") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &cfg->sliced_bucket_count)) return 0;
        } else if (std::strcmp(argv[i], "--single-device") == 0) {
            cfg->use_all_devices = false;
        } else if (std::strcmp(argv[i], "--with-blocked-analysis") == 0) {
            cfg->run_blocked_analysis = true;
        } else if (std::strcmp(argv[i], "--with-finalize") == 0) {
            cfg->run_finalize = true;
        } else if (std::strcmp(argv[i], "--with-python-reference") == 0) {
            cfg->run_python_reference = true;
        } else if (std::strcmp(argv[i], "--no-reuse-artifacts") == 0) {
            cfg->reuse_artifacts = false;
        } else {
            return 0;
        }
    }
    if (cfg->run_finalize || cfg->run_python_reference) cfg->run_blocked_analysis = true;
    return !cfg->h5ad_path.empty() && !cfg->output_dir.empty() && cfg->repeats != 0u;
}

static std::string json_escape(const std::string &value) {
    std::ostringstream out;
    for (char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if ((unsigned char) ch < 0x20u) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (unsigned int) (unsigned char) ch << std::dec << std::setfill(' ');
                } else {
                    out << ch;
                }
        }
    }
    return out.str();
}

static std::string shell_quote(const std::string &value) {
    std::string out = "'";
    for (char ch : value) {
        if (ch == '\'') out += "'\"'\"'";
        else out.push_back(ch);
    }
    out.push_back('\'');
    return out;
}

static void ensure_parent(const fs::path &path) {
    if (path.has_parent_path()) fs::create_directories(path.parent_path());
}

template<typename T>
static void write_vector_raw(std::ofstream *out, const std::vector<T> &values) {
    if (!values.empty()) out->write(reinterpret_cast<const char *>(values.data()), (std::streamsize) (values.size() * sizeof(T)));
}

static bool write_analysis_blob(const fs::path &path, const wb::preprocess_analysis_table &analysis, std::string *error) {
    ensure_parent(path);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    const std::uint64_t rows = (std::uint64_t) analysis.rows;
    const std::uint64_t cols = (std::uint64_t) analysis.cols;
    if (!out) {
        if (error != nullptr) *error = "failed to open analysis blob for write";
        return false;
    }
    if (analysis.cell_total_counts.size() != rows
        || analysis.cell_mito_counts.size() != rows
        || analysis.cell_max_counts.size() != rows
        || analysis.cell_detected_genes.size() != rows
        || analysis.cell_keep.size() != rows
        || analysis.gene_sum.size() != cols
        || analysis.gene_sq_sum.size() != cols
        || analysis.gene_detected_cells.size() != cols
        || analysis.gene_keep.size() != cols
        || analysis.gene_flags.size() != cols) {
        if (error != nullptr) *error = "analysis table shape mismatch";
        return false;
    }
    out.write(reinterpret_cast<const char *>(analysis_magic), sizeof(analysis_magic));
    out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
    write_vector_raw(&out, analysis.cell_total_counts);
    write_vector_raw(&out, analysis.cell_mito_counts);
    write_vector_raw(&out, analysis.cell_max_counts);
    write_vector_raw(&out, analysis.cell_detected_genes);
    write_vector_raw(&out, analysis.cell_keep);
    write_vector_raw(&out, analysis.gene_sum);
    write_vector_raw(&out, analysis.gene_sq_sum);
    write_vector_raw(&out, analysis.gene_detected_cells);
    write_vector_raw(&out, analysis.gene_keep);
    write_vector_raw(&out, analysis.gene_flags);
    if (!out.good()) {
        if (error != nullptr) *error = "failed while writing analysis blob";
        return false;
    }
    return true;
}

static bool h5_delete_attr_if_exists(hid_t object, const char *name) {
    if (H5Aexists(object, name) > 0 && H5Adelete(object, name) < 0) return false;
    return true;
}

static bool h5_write_u32_attr(hid_t object, const char *name, std::uint32_t value) {
    hid_t space = (hid_t) -1;
    hid_t attr = (hid_t) -1;
    if (!h5_delete_attr_if_exists(object, name)) return false;
    space = H5Screate(H5S_SCALAR);
    if (space < 0) return false;
    attr = H5Acreate2(object, name, H5T_NATIVE_UINT32, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) goto done;
    if (H5Awrite(attr, H5T_NATIVE_UINT32, &value) < 0) goto done;
    H5Aclose(attr);
    H5Sclose(space);
    return true;
done:
    if (attr >= 0) H5Aclose(attr);
    if (space >= 0) H5Sclose(space);
    return false;
}

static bool h5_write_u64_attr(hid_t object, const char *name, std::uint64_t value) {
    hid_t space = (hid_t) -1;
    hid_t attr = (hid_t) -1;
    if (!h5_delete_attr_if_exists(object, name)) return false;
    space = H5Screate(H5S_SCALAR);
    if (space < 0) return false;
    attr = H5Acreate2(object, name, H5T_NATIVE_UINT64, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) goto done;
    if (H5Awrite(attr, H5T_NATIVE_UINT64, &value) < 0) goto done;
    H5Aclose(attr);
    H5Sclose(space);
    return true;
done:
    if (attr >= 0) H5Aclose(attr);
    if (space >= 0) H5Sclose(space);
    return false;
}

static bool h5_write_string_attr(hid_t object, const char *name, const char *value) {
    hid_t type = (hid_t) -1;
    hid_t space = (hid_t) -1;
    hid_t attr = (hid_t) -1;
    if (!h5_delete_attr_if_exists(object, name)) return false;
    type = H5Tcopy(H5T_C_S1);
    if (type < 0) return false;
    if (H5Tset_size(type, std::strlen(value)) < 0) goto done;
    if (H5Tset_strpad(type, H5T_STR_NULLTERM) < 0) goto done;
    space = H5Screate(H5S_SCALAR);
    if (space < 0) goto done;
    attr = H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) goto done;
    if (H5Awrite(attr, type, value) < 0) goto done;
    H5Aclose(attr);
    H5Sclose(space);
    H5Tclose(type);
    return true;
done:
    if (attr >= 0) H5Aclose(attr);
    if (space >= 0) H5Sclose(space);
    if (type >= 0) H5Tclose(type);
    return false;
}

static bool h5_ensure_group(hid_t file, const char *path) {
    hid_t group = (hid_t) -1;
    if (H5Lexists(file, path, H5P_DEFAULT) > 0) {
        group = H5Gopen2(file, path, H5P_DEFAULT);
    } else {
        group = H5Gcreate2(file, path, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    }
    if (group < 0) return false;
    H5Gclose(group);
    return true;
}

static bool h5_delete_link_if_exists(hid_t file, const char *path) {
    if (H5Lexists(file, path, H5P_DEFAULT) > 0 && H5Ldelete(file, path, H5P_DEFAULT) < 0) return false;
    return true;
}

template<typename T>
static bool h5_write_1d_dataset(hid_t group, const char *name, hid_t native_type, const std::vector<T> &values) {
    hid_t space = (hid_t) -1;
    hid_t dset = (hid_t) -1;
    const hsize_t dim = (hsize_t) values.size();
    if (H5Lexists(group, name, H5P_DEFAULT) > 0 && H5Ldelete(group, name, H5P_DEFAULT) < 0) return false;
    space = H5Screate_simple(1, &dim, nullptr);
    if (space < 0) return false;
    dset = H5Dcreate2(group, name, native_type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0) goto done;
    if (dim != 0 && H5Dwrite(dset, native_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) < 0) goto done;
    if (dim == 0 && H5Dwrite(dset, native_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, nullptr) < 0) goto done;
    H5Dclose(dset);
    H5Sclose(space);
    return true;
done:
    if (dset >= 0) H5Dclose(dset);
    if (space >= 0) H5Sclose(space);
    return false;
}

static bool clone_bucketed_sliced_partition(cs::bucketed_sliced_ell_partition *dst,
                                            const cs::bucketed_sliced_ell_partition *src) {
    if (dst == nullptr || src == nullptr) return false;
    cs::clear(dst);
    cs::init(dst);
    dst->rows = src->rows;
    dst->cols = src->cols;
    dst->nnz = src->nnz;
    dst->segment_count = src->segment_count;
    dst->segments = dst->segment_count != 0u
        ? (cs::sparse::sliced_ell *) std::calloc((std::size_t) dst->segment_count, sizeof(cs::sparse::sliced_ell))
        : nullptr;
    dst->segment_row_offsets = (std::uint32_t *) std::calloc((std::size_t) dst->segment_count + 1u, sizeof(std::uint32_t));
    dst->exec_to_canonical_rows = dst->rows != 0u ? (std::uint32_t *) std::malloc((std::size_t) dst->rows * sizeof(std::uint32_t)) : nullptr;
    dst->canonical_to_exec_rows = dst->rows != 0u ? (std::uint32_t *) std::malloc((std::size_t) dst->rows * sizeof(std::uint32_t)) : nullptr;
    if ((dst->segment_count != 0u && (dst->segments == nullptr || dst->segment_row_offsets == nullptr))
        || (dst->rows != 0u && (dst->exec_to_canonical_rows == nullptr || dst->canonical_to_exec_rows == nullptr))) {
        cs::clear(dst);
        return false;
    }
    if (dst->segment_count != 0u) {
        std::memcpy(dst->segment_row_offsets,
                    src->segment_row_offsets,
                    ((std::size_t) dst->segment_count + 1u) * sizeof(std::uint32_t));
    }
    if (dst->rows != 0u) {
        std::memcpy(dst->exec_to_canonical_rows, src->exec_to_canonical_rows, (std::size_t) dst->rows * sizeof(std::uint32_t));
        std::memcpy(dst->canonical_to_exec_rows, src->canonical_to_exec_rows, (std::size_t) dst->rows * sizeof(std::uint32_t));
    }
    for (std::uint32_t segment = 0u; segment < dst->segment_count; ++segment) {
        const cs::sparse::sliced_ell *src_segment = src->segments + segment;
        cs::sparse::init(dst->segments + segment, src_segment->rows, src_segment->cols, src_segment->nnz);
        if (!cs::sparse::allocate(dst->segments + segment,
                                  src_segment->slice_count,
                                  src_segment->slice_row_offsets,
                                  src_segment->slice_widths)) {
            cs::clear(dst);
            return false;
        }
        const std::size_t slots = (std::size_t) cs::sparse::total_slots(src_segment);
        if (slots != 0u) {
            std::memcpy(dst->segments[segment].col_idx, src_segment->col_idx, slots * sizeof(cs::types::idx_t));
            std::memcpy(dst->segments[segment].val, src_segment->val, slots * sizeof(*src_segment->val));
        }
    }
    return true;
}

static bool build_sliced_artifact_from_blocked(const fs::path &blocked_path,
                                               const fs::path &sliced_path,
                                               const std::string &cache_root,
                                               int device,
                                               std::uint32_t requested_bucket_count,
                                               std::string *error) {
    wb::dataset_summary summary = wb::summarize_dataset_csh5(blocked_path.string());
    cs::sharded<cs::sparse::blocked_ell> matrix;
    cs::shard_storage storage;
    std::vector<std::uint64_t> partition_rows;
    std::vector<std::uint64_t> partition_nnz;
    std::vector<std::uint32_t> partition_axes;
    std::vector<std::uint64_t> partition_aux;
    std::vector<std::uint64_t> partition_row_offsets;
    std::vector<std::uint32_t> partition_dataset_ids;
    std::vector<std::uint32_t> partition_codec_ids;
    std::vector<std::uint64_t> shard_offsets;
    std::vector<std::uint32_t> part_execution_formats;
    std::vector<std::uint32_t> part_block_sizes;
    std::vector<std::uint32_t> part_bucket_counts;
    std::vector<float> part_fill_ratios;
    std::vector<std::uint64_t> part_execution_bytes;
    std::vector<std::uint64_t> part_blocked_ell_bytes;
    std::vector<std::uint64_t> part_bucketed_blocked_ell_bytes;
    std::vector<std::uint32_t> part_slice_counts;
    std::vector<std::uint32_t> part_slice_rows;
    std::vector<std::uint64_t> part_sliced_bytes;
    std::vector<std::uint64_t> part_bucketed_sliced_bytes;
    std::vector<std::uint32_t> shard_execution_formats;
    std::vector<std::uint32_t> shard_block_sizes;
    std::vector<std::uint32_t> shard_bucketed_partition_counts;
    std::vector<std::uint32_t> shard_bucketed_segment_counts;
    std::vector<float> shard_fill_ratios;
    std::vector<std::uint64_t> shard_execution_bytes;
    std::vector<std::uint64_t> shard_bucketed_blocked_ell_bytes;
    std::vector<std::uint32_t> shard_slice_counts;
    std::vector<std::uint32_t> shard_slice_rows;
    std::vector<std::uint64_t> shard_bucketed_sliced_bytes;
    std::vector<std::uint32_t> shard_pair_ids;
    std::vector<cs::bucketed_sliced_ell_partition> exec_parts;
    hid_t file = (hid_t) -1;
    hid_t matrix_group = (hid_t) -1;
    hid_t codecs_group = (hid_t) -1;
    bool ok = false;

    if (!summary.ok) {
        if (error != nullptr) *error = "failed to summarize blocked input";
        return false;
    }
    if (!fs::exists(blocked_path)) {
        if (error != nullptr) *error = "blocked input dataset does not exist";
        return false;
    }
    fs::copy_file(blocked_path, sliced_path, fs::copy_options::overwrite_existing);
    {
        hid_t sliced_file = H5Fopen(sliced_path.string().c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        if (sliced_file < 0) {
            if (error != nullptr) *error = "failed to open sliced artifact clone for payload setup";
            return false;
        }
        const bool group_ok = h5_ensure_group(sliced_file, "/payload/sliced_ell");
        H5Fclose(sliced_file);
        if (!group_ok) {
            if (error != nullptr) *error = "failed to create sliced payload group";
            return false;
        }
    }

    cs::init(&matrix);
    cs::init(&storage);
    if (!cs::load_dataset_blocked_ell_h5_header(blocked_path.string().c_str(), &matrix, &storage)) {
        if (error != nullptr) *error = "failed to load blocked header while building sliced artifact";
        goto done;
    }

    partition_rows.reserve(summary.partitions.size());
    partition_nnz.reserve(summary.partitions.size());
    partition_axes.reserve(summary.partitions.size());
    partition_aux.assign(summary.partitions.size(), 0u);
    partition_row_offsets.reserve(summary.partitions.size() + 1u);
    partition_dataset_ids.reserve(summary.partitions.size());
    partition_codec_ids.assign(summary.partitions.size(), 1u);
    part_execution_formats.assign(summary.partitions.size(), cs::dataset_execution_format_bucketed_sliced_ell);
    part_block_sizes.assign(summary.partitions.size(), 0u);
    part_bucket_counts.assign(summary.partitions.size(), 0u);
    part_fill_ratios.assign(summary.partitions.size(), 0.0f);
    part_execution_bytes.assign(summary.partitions.size(), 0u);
    part_blocked_ell_bytes.assign(summary.partitions.size(), 0u);
    part_bucketed_blocked_ell_bytes.assign(summary.partitions.size(), 0u);
    part_slice_counts.assign(summary.partitions.size(), 0u);
    part_slice_rows.assign(summary.partitions.size(), 0u);
    part_sliced_bytes.assign(summary.partitions.size(), 0u);
    part_bucketed_sliced_bytes.assign(summary.partitions.size(), 0u);
    exec_parts.resize(summary.partitions.size());
    for (std::size_t i = 0; i < exec_parts.size(); ++i) cs::init(exec_parts.data() + i);

    for (const wb::dataset_partition_summary &part : summary.partitions) {
        partition_rows.push_back(part.rows);
        partition_nnz.push_back(part.nnz);
        partition_axes.push_back(part.axis);
        partition_dataset_ids.push_back(part.dataset_id);
    }
    partition_row_offsets.push_back(0u);
    for (const wb::dataset_partition_summary &part : summary.partitions) partition_row_offsets.push_back(part.row_end);

    shard_offsets.push_back(0u);
    for (const wb::dataset_shard_summary &shard : summary.shards) {
        if (shard_offsets.back() != shard.row_begin) shard_offsets.back() = shard.row_begin;
        shard_offsets.push_back(shard.row_end);
    }

    for (std::size_t part_index = 0; part_index < summary.partitions.size(); ++part_index) {
        cs::sparse::sliced_ell sliced_part;
        cs::bucketed_sliced_ell_partition exec_part;
        std::uint64_t bucketed_bytes = 0u;
        cs::sparse::init(&sliced_part);
        cs::init(&exec_part);
        if (!cs::fetch_dataset_blocked_ell_h5_partition(&matrix, &storage, (unsigned long) part_index)
            || matrix.parts[part_index] == nullptr) {
            if (error != nullptr) *error = "failed to fetch blocked partition while building sliced artifact";
            cs::sparse::clear(&sliced_part);
            cs::clear(&exec_part);
            goto done;
        }
        {
            static const unsigned int slice_candidates[1] = { 32u };
            if (!csc::sliced_ell_from_blocked_ell_cuda_auto(matrix.parts[part_index],
                                                            slice_candidates,
                                                            1u,
                                                            &sliced_part,
                                                            device,
                                                            (cudaStream_t) 0)) {
                if (error != nullptr) *error = "failed to convert blocked partition to sliced-ELL";
                cs::sparse::clear(&sliced_part);
                cs::clear(&exec_part);
                goto done;
            }
        }
        partition_aux[part_index] = cs::partition_aux(&sliced_part);
        part_slice_counts[part_index] = sliced_part.slice_count;
        part_slice_rows[part_index] = cs::sparse::uniform_slice_rows(&sliced_part);
        part_sliced_bytes[part_index] = (std::uint64_t) cs::sparse::bytes(&sliced_part);
        if (!cs::append_sliced_ell_partition_h5(sliced_path.string().c_str(), (unsigned long) part_index, &sliced_part)) {
            if (error != nullptr) *error = "failed to append sliced canonical partition";
            cs::sparse::clear(&sliced_part);
            cs::clear(&exec_part);
            goto done;
        }
        if (!cs::build_bucketed_sliced_ell_partition(&exec_part,
                                                     &sliced_part,
                                                     std::max<std::uint32_t>(1u, requested_bucket_count),
                                                     &bucketed_bytes)) {
            if (error != nullptr) *error = "failed to build sliced execution partition";
            cs::sparse::clear(&sliced_part);
            cs::clear(&exec_part);
            goto done;
        }
        part_bucketed_sliced_bytes[part_index] = bucketed_bytes;
        part_execution_bytes[part_index] = bucketed_bytes != 0u ? bucketed_bytes : part_sliced_bytes[part_index];
        if (!clone_bucketed_sliced_partition(exec_parts.data() + part_index, &exec_part)) {
            if (error != nullptr) *error = "failed to clone sliced execution partition";
            cs::sparse::clear(&sliced_part);
            cs::clear(&exec_part);
            goto done;
        }
        cs::sparse::clear(&sliced_part);
        cs::clear(&exec_part);
    }

    shard_execution_formats.assign(summary.shards.size(), cs::dataset_execution_format_bucketed_sliced_ell);
    shard_block_sizes.assign(summary.shards.size(), 0u);
    shard_bucketed_partition_counts.assign(summary.shards.size(), 0u);
    shard_bucketed_segment_counts.assign(summary.shards.size(), 0u);
    shard_fill_ratios.assign(summary.shards.size(), 0.0f);
    shard_execution_bytes.assign(summary.shards.size(), 0u);
    shard_bucketed_blocked_ell_bytes.assign(summary.shards.size(), 0u);
    shard_slice_counts.assign(summary.shards.size(), 0u);
    shard_slice_rows.assign(summary.shards.size(), 0u);
    shard_bucketed_sliced_bytes.assign(summary.shards.size(), 0u);
    shard_pair_ids.assign(summary.shards.size(), 0u);

    for (std::size_t shard_index = 0; shard_index < summary.shards.size(); ++shard_index) {
        const wb::dataset_shard_summary &shard = summary.shards[shard_index];
        cs::bucketed_sliced_ell_shard exec_shard;
        std::uint64_t shard_rows = 0u;
        std::uint64_t shard_nnz = 0u;
        std::uint32_t uniform_rows = 0u;
        bool same_uniform_rows = true;
        cs::init(&exec_shard);
        exec_shard.partition_count = (std::uint32_t) (shard.partition_end - shard.partition_begin);
        exec_shard.rows = (std::uint32_t) (shard.row_end - shard.row_begin);
        exec_shard.cols = (std::uint32_t) summary.cols;
        exec_shard.nnz = 0u;
        exec_shard.partitions = exec_shard.partition_count != 0u
            ? (cs::bucketed_sliced_ell_partition *) std::calloc((std::size_t) exec_shard.partition_count, sizeof(cs::bucketed_sliced_ell_partition))
            : nullptr;
        exec_shard.partition_row_offsets = (std::uint32_t *) std::calloc((std::size_t) exec_shard.partition_count + 1u, sizeof(std::uint32_t));
        if ((exec_shard.partition_count != 0u && (exec_shard.partitions == nullptr || exec_shard.partition_row_offsets == nullptr))) {
            cs::clear(&exec_shard);
            if (error != nullptr) *error = "failed to allocate sliced execution shard";
            goto done;
        }
        for (std::uint32_t local = 0u; local < exec_shard.partition_count; ++local) {
            const std::size_t global_part = (std::size_t) shard.partition_begin + local;
            if (!clone_bucketed_sliced_partition(exec_shard.partitions + local, exec_parts.data() + global_part)) {
                cs::clear(&exec_shard);
                if (error != nullptr) *error = "failed to populate sliced execution shard";
                goto done;
            }
            exec_shard.partition_row_offsets[local] = (std::uint32_t) shard_rows;
            shard_rows += summary.partitions[global_part].rows;
            shard_nnz += summary.partitions[global_part].nnz;
            exec_shard.nnz += (std::uint32_t) summary.partitions[global_part].nnz;
            shard_execution_bytes[shard_index] += part_execution_bytes[global_part];
            shard_bucketed_sliced_bytes[shard_index] += part_bucketed_sliced_bytes[global_part];
            shard_slice_counts[shard_index] += part_slice_counts[global_part];
            shard_bucketed_partition_counts[shard_index] += 1u;
            shard_bucketed_segment_counts[shard_index] += exec_parts[global_part].segment_count;
            if (local == 0u) uniform_rows = part_slice_rows[global_part];
            else if (uniform_rows != part_slice_rows[global_part]) same_uniform_rows = false;
        }
        exec_shard.partition_row_offsets[exec_shard.partition_count] = (std::uint32_t) shard_rows;
        shard_slice_rows[shard_index] = same_uniform_rows ? uniform_rows : 0u;
        shard_pair_ids[shard_index] = shard.preferred_pair;
        if (!cs::append_bucketed_sliced_ell_shard_h5(sliced_path.string().c_str(), (unsigned long) shard_index, &exec_shard)) {
            cs::clear(&exec_shard);
            if (error != nullptr) *error = "failed to append sliced execution shard";
            goto done;
        }
        cs::clear(&exec_shard);
    }

    file = H5Fopen(sliced_path.string().c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) {
        if (error != nullptr) *error = "failed to reopen sliced artifact for metadata rewrite";
        goto done;
    }
    if (!h5_delete_link_if_exists(file, "/matrix")
        || !h5_delete_link_if_exists(file, "/codecs")
        || !h5_delete_link_if_exists(file, "/execution")
        || !h5_delete_link_if_exists(file, "/payload/blocked_ell")
        || !h5_delete_link_if_exists(file, "/payload/optimized_blocked_ell")) {
        if (error != nullptr) *error = "failed to clear blocked-specific groups from sliced artifact";
        goto done;
    }
    if (!h5_write_string_attr(file, "matrix_format", "sliced_ell")
        || !h5_write_string_attr(file, "payload_layout", "optimized_bucketed_sliced_ell")
        || !h5_write_u64_attr(file, "num_codecs", 1u)
        || !h5_write_u64_attr(file, "num_partitions", (std::uint64_t) summary.partitions.size())
        || !h5_write_u64_attr(file, "num_shards", (std::uint64_t) summary.shards.size())) {
        if (error != nullptr) *error = "failed to rewrite root attributes for sliced artifact";
        goto done;
    }

    matrix_group = H5Gcreate2(file, "/matrix", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    codecs_group = H5Gcreate2(file, "/codecs", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (matrix_group < 0 || codecs_group < 0) {
        if (error != nullptr) *error = "failed to recreate matrix/codecs groups in sliced artifact";
        goto done;
    }
    if (!h5_write_1d_dataset(matrix_group, "partition_rows", H5T_NATIVE_UINT64, partition_rows)
        || !h5_write_1d_dataset(matrix_group, "partition_nnz", H5T_NATIVE_UINT64, partition_nnz)
        || !h5_write_1d_dataset(matrix_group, "partition_axes", H5T_NATIVE_UINT32, partition_axes)
        || !h5_write_1d_dataset(matrix_group, "partition_aux", H5T_NATIVE_UINT64, partition_aux)
        || !h5_write_1d_dataset(matrix_group, "partition_row_offsets", H5T_NATIVE_UINT64, partition_row_offsets)
        || !h5_write_1d_dataset(matrix_group, "partition_dataset_ids", H5T_NATIVE_UINT32, partition_dataset_ids)
        || !h5_write_1d_dataset(matrix_group, "partition_codec_ids", H5T_NATIVE_UINT32, partition_codec_ids)
        || !h5_write_1d_dataset(matrix_group, "shard_offsets", H5T_NATIVE_UINT64, shard_offsets)) {
        if (error != nullptr) *error = "failed to rewrite sliced matrix layout metadata";
        goto done;
    }
    {
        std::vector<std::uint32_t> codec_id{1u};
        std::vector<std::uint32_t> family{cs::dataset_codec_family_sliced_ell};
        std::vector<std::uint32_t> value_code{0u};
        std::vector<std::uint32_t> scale_value_code{0u};
        std::vector<std::uint32_t> bits{16u};
        std::vector<std::uint32_t> flags{0u};
        if (!h5_write_1d_dataset(codecs_group, "codec_id", H5T_NATIVE_UINT32, codec_id)
            || !h5_write_1d_dataset(codecs_group, "family", H5T_NATIVE_UINT32, family)
            || !h5_write_1d_dataset(codecs_group, "value_code", H5T_NATIVE_UINT32, value_code)
            || !h5_write_1d_dataset(codecs_group, "scale_value_code", H5T_NATIVE_UINT32, scale_value_code)
            || !h5_write_1d_dataset(codecs_group, "bits", H5T_NATIVE_UINT32, bits)
            || !h5_write_1d_dataset(codecs_group, "flags", H5T_NATIVE_UINT32, flags)) {
            if (error != nullptr) *error = "failed to rewrite sliced codec metadata";
            goto done;
        }
    }
    H5Gclose(codecs_group);
    codecs_group = (hid_t) -1;
    H5Gclose(matrix_group);
    matrix_group = (hid_t) -1;
    H5Fclose(file);
    file = (hid_t) -1;

    {
        cs::dataset_execution_view execution{};
        execution.partition_count = (std::uint32_t) part_execution_formats.size();
        execution.partition_execution_formats = part_execution_formats.data();
        execution.partition_blocked_ell_block_sizes = part_block_sizes.data();
        execution.partition_blocked_ell_bucket_counts = part_bucket_counts.data();
        execution.partition_blocked_ell_fill_ratios = part_fill_ratios.data();
        execution.partition_execution_bytes = part_execution_bytes.data();
        execution.partition_blocked_ell_bytes = part_blocked_ell_bytes.data();
        execution.partition_bucketed_blocked_ell_bytes = part_bucketed_blocked_ell_bytes.data();
        execution.partition_sliced_ell_slice_counts = part_slice_counts.data();
        execution.partition_sliced_ell_slice_rows = part_slice_rows.data();
        execution.partition_sliced_ell_bytes = part_sliced_bytes.data();
        execution.partition_bucketed_sliced_ell_bytes = part_bucketed_sliced_bytes.data();
        execution.shard_count = (std::uint32_t) shard_execution_formats.size();
        execution.shard_execution_formats = shard_execution_formats.data();
        execution.shard_blocked_ell_block_sizes = shard_block_sizes.data();
        execution.shard_bucketed_partition_counts = shard_bucketed_partition_counts.data();
        execution.shard_bucketed_segment_counts = shard_bucketed_segment_counts.data();
        execution.shard_blocked_ell_fill_ratios = shard_fill_ratios.data();
        execution.shard_execution_bytes = shard_execution_bytes.data();
        execution.shard_bucketed_blocked_ell_bytes = shard_bucketed_blocked_ell_bytes.data();
        execution.shard_sliced_ell_slice_counts = shard_slice_counts.data();
        execution.shard_sliced_ell_slice_rows = shard_slice_rows.data();
        execution.shard_bucketed_sliced_ell_bytes = shard_bucketed_sliced_bytes.data();
        execution.shard_preferred_pair_ids = shard_pair_ids.data();
        execution.preferred_base_format = cs::dataset_execution_format_bucketed_sliced_ell;
        if (!cs::append_dataset_execution_h5(sliced_path.string().c_str(), &execution)) {
            if (error != nullptr) *error = "failed to append sliced execution metadata";
            goto done;
        }
    }
    if (!cache_root.empty() && !cs::warm_dataset_sliced_ell_h5_cache(sliced_path.string().c_str(), cache_root.c_str())) {
        if (error != nullptr) *error = "failed to warm sliced cache";
        goto done;
    }
    ok = true;

done:
    for (cs::bucketed_sliced_ell_partition &part : exec_parts) cs::clear(&part);
    if (codecs_group >= 0) H5Gclose(codecs_group);
    if (matrix_group >= 0) H5Gclose(matrix_group);
    if (file >= 0) H5Fclose(file);
    cs::clear(&storage);
    cs::clear(&matrix);
    if (!ok) {
        std::error_code ec;
        fs::remove(sliced_path, ec);
    }
    return ok;
}

static double mean_ms(const std::vector<double> &values) {
    if (values.empty()) return 0.0;
    double total = 0.0;
    for (double value : values) total += value;
    return total / (double) values.size();
}

static std::map<std::string, std::string> parse_summary_tsv(const fs::path &path) {
    std::ifstream in(path);
    std::map<std::string, std::string> values;
    std::string line;
    while (std::getline(in, line)) {
        const std::size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        values.emplace(line.substr(0, tab), line.substr(tab + 1u));
    }
    return values;
}

static bool parse_numeric_string(const std::string &value, double *out) {
    char *end = nullptr;
    const double parsed = std::strtod(value.c_str(), &end);
    if (value.empty() || end == value.c_str() || *end != 0) return false;
    *out = parsed;
    return true;
}

static bool run_python_reference(const config &cfg,
                                 const dataset_artifacts &artifacts,
                                 std::map<std::string, std::string> *summary_out,
                                 std::string *error) {
    const fs::path script_path = fs::path(__FILE__).parent_path() / "preprocess_scanpy_reference.py";
    const std::string cmd =
        shell_quote(cfg.python_exe) + " "
        + shell_quote(script_path.string())
        + " --h5ad " + shell_quote(cfg.h5ad_path)
        + " --matrix-source " + shell_quote(cfg.matrix_source)
        + " --blocked-analysis " + shell_quote(artifacts.blocked_analysis_blob.string())
        + " --sliced-analysis " + shell_quote(artifacts.sliced_analysis_blob.string())
        + " --summary-tsv " + shell_quote(artifacts.stability_summary_tsv.string())
        + " --details-json " + shell_quote(artifacts.stability_details_json.string())
        + " --target-sum " + std::to_string(cfg.target_sum)
        + " --min-counts " + std::to_string(cfg.min_counts)
        + " --min-genes " + std::to_string(cfg.min_genes)
        + " --max-mito-fraction " + std::to_string(cfg.max_mito_fraction)
        + " --min-gene-sum " + std::to_string(cfg.min_gene_sum)
        + " --min-detected-cells " + std::to_string(cfg.min_detected_cells)
        + " --min-variance " + std::to_string(cfg.min_variance);
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        if (error != nullptr) *error = "python stability comparison failed";
        return false;
    }
    if (summary_out != nullptr) *summary_out = parse_summary_tsv(artifacts.stability_summary_tsv);
    return true;
}

static wb::preprocess_config make_preprocess_config(const config &cfg) {
    wb::preprocess_config preprocess;
    preprocess.target_sum = cfg.target_sum;
    preprocess.min_counts = cfg.min_counts;
    preprocess.min_genes = cfg.min_genes;
    preprocess.max_mito_fraction = cfg.max_mito_fraction;
    preprocess.min_gene_sum = cfg.min_gene_sum;
    preprocess.min_detected_cells = cfg.min_detected_cells;
    preprocess.min_variance = cfg.min_variance;
    preprocess.device = cfg.device;
    preprocess.use_all_devices = cfg.use_all_devices;
    preprocess.finalize_after_preprocess = cfg.run_finalize;
    preprocess.cache_dir = cfg.cache_root;
    preprocess.working_root = cfg.working_root;
    return preprocess;
}

static void write_run_config(const config &cfg,
                             const dataset_artifacts &artifacts,
                             const wb::manifest_inspection &inspection) {
    ensure_parent(artifacts.run_config_json);
    std::ofstream out(artifacts.run_config_json, std::ios::trunc);
    out << "{\n";
    out << "  \"h5ad\": \"" << json_escape(cfg.h5ad_path) << "\",\n";
    out << "  \"matrix_source\": \"" << json_escape(cfg.matrix_source) << "\",\n";
    out << "  \"output_dir\": \"" << json_escape(cfg.output_dir) << "\",\n";
    out << "  \"cache_root\": \"" << json_escape(cfg.cache_root) << "\",\n";
    out << "  \"working_root\": \"" << json_escape(cfg.working_root) << "\",\n";
    out << "  \"python_exe\": \"" << json_escape(cfg.python_exe) << "\",\n";
    out << "  \"warmup\": " << cfg.warmup << ",\n";
    out << "  \"repeats\": " << cfg.repeats << ",\n";
    out << "  \"device\": " << cfg.device << ",\n";
    out << "  \"use_all_devices\": " << (cfg.use_all_devices ? "true" : "false") << ",\n";
    out << "  \"reader_bytes\": " << cfg.reader_bytes << ",\n";
    out << "  \"max_part_nnz\": " << cfg.max_part_nnz << ",\n";
    out << "  \"convert_window_bytes\": " << cfg.convert_window_bytes << ",\n";
    out << "  \"target_shard_bytes\": " << cfg.target_shard_bytes << ",\n";
    out << "  \"target_sum\": " << cfg.target_sum << ",\n";
    out << "  \"min_counts\": " << cfg.min_counts << ",\n";
    out << "  \"min_genes\": " << cfg.min_genes << ",\n";
    out << "  \"max_mito_fraction\": " << cfg.max_mito_fraction << ",\n";
    out << "  \"min_gene_sum\": " << cfg.min_gene_sum << ",\n";
    out << "  \"min_detected_cells\": " << cfg.min_detected_cells << ",\n";
    out << "  \"min_variance\": " << cfg.min_variance << ",\n";
    out << "  \"sliced_bucket_count\": " << cfg.sliced_bucket_count << ",\n";
    out << "  \"run_blocked_analysis\": " << (cfg.run_blocked_analysis ? "true" : "false") << ",\n";
    out << "  \"run_finalize\": " << (cfg.run_finalize ? "true" : "false") << ",\n";
    out << "  \"run_python_reference\": " << (cfg.run_python_reference ? "true" : "false") << ",\n";
    out << "  \"use_all_devices\": " << (cfg.use_all_devices ? "true" : "false") << ",\n";
    out << "  \"reuse_artifacts\": " << (cfg.reuse_artifacts ? "true" : "false") << ",\n";
    out << "  \"artifacts\": {\n";
    out << "    \"blocked_input\": \"" << json_escape(artifacts.blocked_input.string()) << "\",\n";
    out << "    \"sliced_input\": \"" << json_escape(artifacts.sliced_input.string()) << "\",\n";
    out << "    \"blocked_final\": \"" << json_escape(artifacts.blocked_final.string()) << "\"\n";
    out << "  },\n";
    out << "  \"inspected_rows\": " << (inspection.sources.empty() ? 0ul : inspection.sources.front().rows) << ",\n";
    out << "  \"inspected_cols\": " << (inspection.sources.empty() ? 0ul : inspection.sources.front().cols) << ",\n";
    out << "  \"inspected_nnz\": " << (inspection.sources.empty() ? 0ul : inspection.sources.front().nnz) << "\n";
    out << "}\n";
}

static void write_results(const dataset_artifacts &artifacts,
                          const config &cfg,
                          const wb::dataset_summary &blocked_input,
                          const wb::dataset_summary &sliced_input,
                          const wb::dataset_summary &blocked_final,
                          const std::vector<double> &sliced_ms,
                          const std::vector<double> &blocked_ms,
                          const std::vector<double> &finalize_ms,
                          const std::vector<double> &browse_ms,
                          double materialize_blocked_ms,
                          double materialize_sliced_ms,
                          double python_ms,
                          double end_to_end_ms,
                          const std::map<std::string, std::string> &stability) {
    const std::uint64_t blocked_input_file_bytes = fs::exists(artifacts.blocked_input) ? (std::uint64_t) fs::file_size(artifacts.blocked_input) : 0u;
    const std::uint64_t sliced_input_file_bytes = fs::exists(artifacts.sliced_input) ? (std::uint64_t) fs::file_size(artifacts.sliced_input) : 0u;
    const std::uint64_t blocked_final_file_bytes = fs::exists(artifacts.blocked_final) ? (std::uint64_t) fs::file_size(artifacts.blocked_final) : 0u;
    std::ofstream out(artifacts.results_json, std::ios::trunc);
    out << "{\n";
    out << "  \"artifacts\": {\n";
    out << "    \"blocked_input\": \"" << json_escape(artifacts.blocked_input.string()) << "\",\n";
    out << "    \"sliced_input\": \"" << json_escape(artifacts.sliced_input.string()) << "\",\n";
    out << "    \"blocked_final\": \"" << json_escape(artifacts.blocked_final.string()) << "\"\n";
    out << "  },\n";
    out << "  \"options\": {\n";
    out << "    \"run_blocked_analysis\": " << (cfg.run_blocked_analysis ? "true" : "false") << ",\n";
    out << "    \"run_finalize\": " << (cfg.run_finalize ? "true" : "false") << ",\n";
    out << "    \"run_python_reference\": " << (cfg.run_python_reference ? "true" : "false") << ",\n";
    out << "    \"use_all_devices\": " << (cfg.use_all_devices ? "true" : "false") << ",\n";
    out << "    \"reuse_artifacts\": " << (cfg.reuse_artifacts ? "true" : "false") << "\n";
    out << "  },\n";
    out << "  \"dataset\": {\n";
    out << "    \"rows\": " << blocked_input.rows << ",\n";
    out << "    \"cols\": " << blocked_input.cols << ",\n";
    out << "    \"nnz\": " << blocked_input.nnz << ",\n";
    out << "    \"partitions\": " << blocked_input.num_partitions << ",\n";
    out << "    \"shards\": " << blocked_input.num_shards << "\n";
    out << "  },\n";
  out << "  \"formats\": {\n";
    out << "    \"blocked_input\": \"" << json_escape(blocked_input.matrix_format) << "\",\n";
    out << "    \"sliced_input\": \"" << json_escape(sliced_input.matrix_format) << "\",\n";
    out << "    \"blocked_final\": \"" << json_escape(blocked_final.matrix_format) << "\"\n";
    out << "  },\n";
    out << "  \"artifact_file_bytes\": {\n";
    out << "    \"blocked_input\": " << blocked_input_file_bytes << ",\n";
    out << "    \"sliced_input\": " << sliced_input_file_bytes << ",\n";
    out << "    \"blocked_final\": " << blocked_final_file_bytes << "\n";
    out << "  },\n";
    out << "  \"timings_ms\": {\n";
    out << "    \"materialize_blocked\": " << materialize_blocked_ms << ",\n";
    out << "    \"materialize_sliced\": " << materialize_sliced_ms << ",\n";
    out << "    \"analyze_blocked_mean\": " << (cfg.run_blocked_analysis ? mean_ms(blocked_ms) : 0.0) << ",\n";
    out << "    \"analyze_sliced_mean\": " << mean_ms(sliced_ms) << ",\n";
    out << "    \"finalize_blocked_mean\": " << mean_ms(finalize_ms) << ",\n";
    out << "    \"browse_blocked_mean\": " << mean_ms(browse_ms) << ",\n";
    out << "    \"python_reference\": " << python_ms << ",\n";
    out << "    \"end_to_end\": " << end_to_end_ms << "\n";
    out << "  },\n";
    out << "  \"stability\": {\n";
    bool first = true;
    for (const auto &entry : stability) {
        if (!first) out << ",\n";
        first = false;
        double numeric = 0.0;
        out << "    \"" << json_escape(entry.first) << "\": ";
        if (parse_numeric_string(entry.second, &numeric)) out << numeric;
        else out << "\"" << json_escape(entry.second) << "\"";
    }
    out << "\n  }\n";
    out << "}\n";
}

static void write_summary(const dataset_artifacts &artifacts,
                          const config &cfg,
                          const wb::dataset_summary &blocked_input,
                          const wb::dataset_summary &sliced_input,
                          const wb::dataset_summary &blocked_final,
                          const std::vector<double> &sliced_ms,
                          const std::vector<double> &blocked_ms,
                          const std::vector<double> &finalize_ms,
                          const std::vector<double> &browse_ms,
                          const std::map<std::string, std::string> &stability) {
    {
        std::ofstream out(artifacts.summary_json, std::ios::trunc);
        out << "{\n";
        out << "  \"blocked_input_rows\": " << blocked_input.rows << ",\n";
        out << "  \"blocked_final_rows\": " << blocked_final.rows << ",\n";
        out << "  \"sliced_input_rows\": " << sliced_input.rows << ",\n";
        out << "  \"blocked_analyze_mean_ms\": " << (cfg.run_blocked_analysis ? mean_ms(blocked_ms) : 0.0) << ",\n";
        out << "  \"sliced_analyze_mean_ms\": " << mean_ms(sliced_ms) << ",\n";
        out << "  \"blocked_finalize_mean_ms\": " << mean_ms(finalize_ms) << ",\n";
        out << "  \"browse_mean_ms\": " << mean_ms(browse_ms) << ",\n";
        out << "  \"run_blocked_analysis\": " << (cfg.run_blocked_analysis ? "true" : "false") << ",\n";
        out << "  \"run_finalize\": " << (cfg.run_finalize ? "true" : "false") << ",\n";
        out << "  \"use_all_devices\": " << (cfg.use_all_devices ? "true" : "false") << ",\n";
        out << "  \"run_python_reference\": " << (cfg.run_python_reference ? "true" : "false") << ",\n";
        out << "  \"baseline_mode\": \"" << json_escape(stability.count("baseline_mode") != 0 ? stability.at("baseline_mode") : std::string()) << "\"\n";
        out << "}\n";
    }
    {
        std::ofstream out(artifacts.summary_txt, std::ios::trunc);
        out << "blocked_input: " << blocked_input.rows << " rows, " << blocked_input.cols << " cols, " << blocked_input.nnz << " nnz\n";
        out << "sliced_input: " << sliced_input.rows << " rows, " << sliced_input.cols << " cols, " << sliced_input.nnz << " nnz\n";
        if (cfg.run_finalize) out << "blocked_final: " << blocked_final.rows << " rows, " << blocked_final.cols << " cols, " << blocked_final.nnz << " nnz\n";
        else out << "blocked_final: skipped\n";
        out << "use_all_devices: " << (cfg.use_all_devices ? "true" : "false") << "\n";
        out << "analyze_blocked_mean_ms: " << (cfg.run_blocked_analysis ? mean_ms(blocked_ms) : 0.0) << "\n";
        out << "analyze_sliced_mean_ms: " << mean_ms(sliced_ms) << "\n";
        out << "finalize_blocked_mean_ms: " << (cfg.run_finalize ? mean_ms(finalize_ms) : 0.0) << "\n";
        out << "browse_mean_ms: " << (cfg.run_finalize ? mean_ms(browse_ms) : 0.0) << "\n";
        const auto it = stability.find("baseline_mode");
        out << "baseline_mode: " << (cfg.run_python_reference && it != stability.end() ? it->second : std::string("skipped")) << "\n";
        auto emit = [&](const char *key) {
            const auto found = stability.find(key);
            if (found != stability.end()) out << key << ": " << found->second << "\n";
        };
        if (cfg.run_blocked_analysis) {
            emit("blocked.cell_keep.mismatch_count");
            emit("blocked.gene_keep.mismatch_count");
            emit("blocked.gene_sum.max_abs");
        }
        emit("sliced.cell_keep.mismatch_count");
        emit("sliced.gene_keep.mismatch_count");
        emit("sliced.gene_sum.max_abs");
    }
}

int benchmark_main(int argc, char **argv) {
    config cfg;
    dataset_artifacts artifacts;
    wb::manifest_inspection inspection;
    wb::ingest_plan plan;
    wb::conversion_report conversion;
    wb::dataset_summary blocked_input_summary;
    wb::dataset_summary sliced_input_summary;
    wb::dataset_summary blocked_final_summary;
    std::vector<double> blocked_analyze_ms;
    std::vector<double> sliced_analyze_ms;
    std::vector<double> finalize_stage_ms;
    std::vector<double> browse_stage_ms;
    std::map<std::string, std::string> stability_summary;
    double materialize_blocked_ms = 0.0;
    double materialize_sliced_ms = 0.0;
    double python_ms = 0.0;
    const auto end_to_end_begin = std::chrono::steady_clock::now();

    if (!parse_args(argc, argv, &cfg)) {
        usage(argv[0]);
        return 1;
    }

    if (cfg.cache_root.empty()) cfg.cache_root = (fs::path(cfg.output_dir) / "cache").string();
    if (cfg.working_root.empty()) cfg.working_root = (fs::path(cfg.output_dir) / "work").string();
    fs::create_directories(cfg.output_dir);
    fs::create_directories(cfg.cache_root);
    fs::create_directories(cfg.working_root);

    artifacts.blocked_input = fs::path(cfg.output_dir) / "input.blocked.csh5";
    artifacts.sliced_input = fs::path(cfg.output_dir) / "input.sliced.csh5";
    artifacts.blocked_final = fs::path(cfg.output_dir) / "final.blocked.csh5";
    artifacts.blocked_analysis_blob = fs::path(cfg.output_dir) / "analysis.blocked.bin";
    artifacts.sliced_analysis_blob = fs::path(cfg.output_dir) / "analysis.sliced.bin";
    artifacts.stability_summary_tsv = fs::path(cfg.output_dir) / "stability.summary.tsv";
    artifacts.stability_details_json = fs::path(cfg.output_dir) / "stability.details.json";
    artifacts.run_config_json = fs::path(cfg.output_dir) / "run_config.json";
    artifacts.results_json = fs::path(cfg.output_dir) / "results.json";
    artifacts.summary_json = fs::path(cfg.output_dir) / "summary.json";
    artifacts.summary_txt = fs::path(cfg.output_dir) / "summary.txt";

    int device_ids[16];
    std::size_t device_count = 0u;
    if (cfg.use_all_devices) {
        int visible = 0;
        if (cudaGetDeviceCount(&visible) == cudaSuccess && visible > 0) {
            for (int device = 0; device < visible && device_count < (sizeof(device_ids) / sizeof(device_ids[0])); ++device) {
                device_ids[device_count++] = device;
            }
        }
    }
    if (device_count == 0u) {
        device_ids[0] = cfg.device;
        device_count = 1u;
    }
    cellerator::bench::benchmark_mutex_guard benchmark_mutex("realPreprocessBench", device_ids, device_count);

    const bool can_reuse_sliced_input = cfg.reuse_artifacts && fs::exists(artifacts.sliced_input);
    if (!can_reuse_sliced_input) {
        scoped_nvtx_range range("load_or_generate");
        wb::source_entry source;
        source.dataset_id = fs::path(cfg.h5ad_path).stem().string();
        source.matrix_path = cfg.h5ad_path;
        source.matrix_source = cfg.matrix_source;
        source.allow_processed = true;
        source.format = ::cellerator::ingest::dataset::source_h5ad;
        inspection = wb::inspect_source_entries({source}, "realPreprocessBench", cfg.reader_bytes);
        if (!inspection.ok || inspection.sources.empty()) {
            std::fprintf(stderr, "realPreprocessBench: failed to inspect %s\n", cfg.h5ad_path.c_str());
            for (const wb::issue &issue : inspection.issues) {
                std::fprintf(stderr, "  [%s] %s: %s\n",
                             wb::severity_name(issue.severity).c_str(),
                             issue.scope.c_str(),
                             issue.message.c_str());
            }
            return 1;
        }
        wb::ingest_policy policy;
        policy.max_part_nnz = cfg.max_part_nnz;
        policy.convert_window_bytes = cfg.convert_window_bytes;
        policy.target_shard_bytes = cfg.target_shard_bytes;
        policy.reader_bytes = cfg.reader_bytes;
        policy.output_path = artifacts.sliced_input.string();
        policy.cache_dir.clear();
        policy.working_root = cfg.working_root;
        policy.device = cfg.device;
        policy.embed_metadata = true;
        policy.build_browse_cache = false;
        plan = wb::plan_dataset_ingest(inspection.sources, policy);
        if (!plan.ok) {
            std::fprintf(stderr, "realPreprocessBench: failed to build ingest plan\n");
            return 1;
        }
    }

    write_run_config(cfg, artifacts, inspection);

    {
        scoped_nvtx_range range("materialize_sliced");
        if (!can_reuse_sliced_input) {
            const auto phase_begin = std::chrono::steady_clock::now();
            conversion = wb::convert_plan_to_dataset_csh5(plan);
            if (!conversion.ok) {
                std::fprintf(stderr, "realPreprocessBench: sliced ingest conversion failed\n");
                for (const wb::issue &issue : conversion.issues) {
                    std::fprintf(stderr, "  [%s] %s: %s\n",
                                 wb::severity_name(issue.severity).c_str(),
                                 issue.scope.c_str(),
                                 issue.message.c_str());
                }
                return 1;
            }
            const auto phase_end = std::chrono::steady_clock::now();
            materialize_sliced_ms = std::chrono::duration<double, std::milli>(phase_end - phase_begin).count();
        }
        sliced_input_summary = wb::summarize_dataset_csh5(artifacts.sliced_input.string());
        if (!sliced_input_summary.ok) {
            std::fprintf(stderr, "realPreprocessBench: failed to summarize sliced input\n");
            return 1;
        }
        if (!cfg.cache_root.empty()
            && !cs::warm_dataset_sliced_ell_h5_cache(artifacts.sliced_input.string().c_str(), cfg.cache_root.c_str())) {
            std::fprintf(stderr, "realPreprocessBench: failed to warm sliced cache\n");
            return 1;
        }
    }
    if (cfg.run_blocked_analysis) {
        if (!fs::exists(artifacts.blocked_input)) {
            std::fprintf(stderr,
                         "realPreprocessBench: blocked compatibility path currently requires an existing %s artifact\n",
                         artifacts.blocked_input.string().c_str());
            return 1;
        }
        blocked_input_summary = wb::summarize_dataset_csh5(artifacts.blocked_input.string());
        if (!blocked_input_summary.ok) {
            std::fprintf(stderr, "realPreprocessBench: failed to summarize blocked input\n");
            return 1;
        }
        if (!cfg.cache_root.empty()) {
            if (!cs::warm_dataset_blocked_ell_h5_cache(artifacts.blocked_input.string().c_str(), cfg.cache_root.c_str())
                || !cs::warm_dataset_blocked_ell_h5_execution_cache(artifacts.blocked_input.string().c_str(), cfg.cache_root.c_str())) {
                std::fprintf(stderr, "realPreprocessBench: failed to warm blocked compatibility cache\n");
                return 1;
            }
        }
    }

    for (unsigned int iter = 0u; iter < cfg.warmup + cfg.repeats; ++iter) {
        const bool measured = iter >= cfg.warmup;
        const unsigned int measured_index = iter - cfg.warmup;
        wb::preprocess_config analyze_cfg = make_preprocess_config(cfg);
        analyze_cfg.finalize_after_preprocess = false;

        wb::preprocess_analysis_table sliced_analysis;
        wb::preprocess_analysis_table blocked_analysis;
        bool have_blocked_analysis = false;

        {
            scoped_nvtx_range range("steady_state_compute.sliced");
            const auto phase_begin = std::chrono::steady_clock::now();
            sliced_analysis = wb::analyze_dataset_preprocess(artifacts.sliced_input.string(), analyze_cfg);
            const auto phase_end = std::chrono::steady_clock::now();
            if (!sliced_analysis.ok) {
                std::fprintf(stderr, "realPreprocessBench: sliced analyze failed\n");
                return 1;
            }
            if (measured) sliced_analyze_ms.push_back(std::chrono::duration<double, std::milli>(phase_end - phase_begin).count());
        }
        if (cfg.run_blocked_analysis) {
            scoped_nvtx_range range("steady_state_compute.blocked");
            const auto phase_begin = std::chrono::steady_clock::now();
            blocked_analysis = wb::analyze_dataset_preprocess(artifacts.blocked_input.string(), analyze_cfg);
            const auto phase_end = std::chrono::steady_clock::now();
            if (!blocked_analysis.ok) {
                std::fprintf(stderr, "realPreprocessBench: blocked analyze failed\n");
                return 1;
            }
            have_blocked_analysis = true;
            if (measured) blocked_analyze_ms.push_back(std::chrono::duration<double, std::milli>(phase_end - phase_begin).count());
        }

        if (measured && measured_index + 1u == cfg.repeats) {
            std::string error;
            if (!write_analysis_blob(artifacts.sliced_analysis_blob, sliced_analysis, &error)
                || (have_blocked_analysis && !write_analysis_blob(artifacts.blocked_analysis_blob, blocked_analysis, &error))) {
                std::fprintf(stderr, "realPreprocessBench: %s\n", error.c_str());
                return 1;
            }
        }

        if (cfg.run_finalize) {
            const fs::path target = measured && measured_index + 1u == cfg.repeats
                ? artifacts.blocked_final
                : (fs::path(cfg.output_dir) / ("scratch_finalize_" + std::to_string(iter) + ".blocked.csh5"));
            wb::preprocess_config finalize_cfg = make_preprocess_config(cfg);
            finalize_cfg.finalize_after_preprocess = true;
            fs::copy_file(artifacts.blocked_input, target, fs::copy_options::overwrite_existing);
            scoped_nvtx_range range("finalize.blocked");
            wb::preprocess_persist_summary persisted;
            persisted = wb::persist_preprocess_analysis(target.string(), blocked_analysis, finalize_cfg);
            if (!persisted.summary.ok) {
                std::fprintf(stderr, "realPreprocessBench: finalize persist failed\n");
                return 1;
            }
            if (measured) {
                finalize_stage_ms.push_back(persisted.persist_ms);
                browse_stage_ms.push_back(persisted.browse_ms);
            } else {
                std::error_code ec;
                fs::remove(target, ec);
            }
        }
    }

    if (cfg.run_finalize) {
        blocked_final_summary = wb::summarize_dataset_csh5(artifacts.blocked_final.string());
        if (!blocked_final_summary.ok) {
            std::fprintf(stderr, "realPreprocessBench: failed to summarize final blocked artifact\n");
            return 1;
        }
    } else {
        blocked_final_summary = blocked_input_summary;
    }

    if (cfg.run_python_reference) {
        scoped_nvtx_range range("python_reference");
        const auto phase_begin = std::chrono::steady_clock::now();
        std::string error;
        if (!run_python_reference(cfg, artifacts, &stability_summary, &error)) {
            std::fprintf(stderr, "realPreprocessBench: %s\n", error.c_str());
            return 1;
        }
        const auto phase_end = std::chrono::steady_clock::now();
        python_ms = std::chrono::duration<double, std::milli>(phase_end - phase_begin).count();
    } else {
        stability_summary["baseline_mode"] = "skipped";
    }

    const auto end_to_end_end = std::chrono::steady_clock::now();
    const double end_to_end_ms = std::chrono::duration<double, std::milli>(end_to_end_end - end_to_end_begin).count();

    write_results(artifacts,
                  cfg,
                  blocked_input_summary,
                  sliced_input_summary,
                  blocked_final_summary,
                  sliced_analyze_ms,
                  blocked_analyze_ms,
                  finalize_stage_ms,
                  browse_stage_ms,
                  materialize_blocked_ms,
                  materialize_sliced_ms,
                  python_ms,
                  end_to_end_ms,
                  stability_summary);
    write_summary(artifacts,
                  cfg,
                  blocked_input_summary,
                  sliced_input_summary,
                  blocked_final_summary,
                  sliced_analyze_ms,
                  blocked_analyze_ms,
                  finalize_stage_ms,
                  browse_stage_ms,
                  stability_summary);

    std::printf("blocked_mean_ms=%.3f sliced_mean_ms=%.3f finalize_mean_ms=%.3f browse_mean_ms=%.3f baseline=%s\n",
                cfg.run_blocked_analysis ? mean_ms(blocked_analyze_ms) : 0.0,
                mean_ms(sliced_analyze_ms),
                cfg.run_finalize ? mean_ms(finalize_stage_ms) : 0.0,
                cfg.run_finalize ? mean_ms(browse_stage_ms) : 0.0,
                stability_summary.count("baseline_mode") != 0 ? stability_summary["baseline_mode"].c_str() : "unknown");
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    return benchmark_main(argc, argv);
}
