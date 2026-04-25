#include "benchmark_mutex.hh"
#include <Cellerator/workbench/dataset_workbench.hh>
#include "../extern/CellShard/include/CellShard/CellShard.hh"

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

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
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace wb = ::cellerator::apps::workbench;
namespace cs = ::cellshard;

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
    bool use_all_devices = true;
    bool run_python_reference = false;
    bool reuse_artifacts = true;
};

struct scoped_nvtx_range {
    explicit scoped_nvtx_range(const char *label) { nvtxRangePushA(label); }
    ~scoped_nvtx_range() { nvtxRangePop(); }
};

struct dataset_artifacts {
    fs::path raw_sliced_input;
    fs::path filtered_sliced_output;
    fs::path analysis_blob;
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
                 "  --working-root PATH         Working root for ingest temp files. Default: OUTPUT/work.\n"
                 "  --matrix-source NAME        H5AD matrix source. Default: raw_x.\n"
                 "  --python-exe PATH           Python interpreter. Default: python3.\n"
                 "  --reader-bytes-mb N         H5AD reader window in MiB. Default: 8.\n"
                 "  --max-part-nnz N            Max nnz per physical partition. Default: 16777216.\n"
                 "  --convert-window-mb N       COO load/convert window in MiB. Default: 256.\n"
                 "  --target-shard-mb N         Target shard bytes in MiB. Default: 512.\n"
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
                 "  --single-device             Disable multi-GPU preprocess and use --device only.\n"
                 "  --with-python-reference     Run Python numerical-stability baseline.\n"
                 "  --no-reuse-artifacts        Rebuild generated artifacts even if present.\n",
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
        } else if (std::strcmp(argv[i], "--single-device") == 0) {
            cfg->use_all_devices = false;
        } else if (std::strcmp(argv[i], "--with-python-reference") == 0) {
            cfg->run_python_reference = true;
        } else if (std::strcmp(argv[i], "--no-reuse-artifacts") == 0) {
            cfg->reuse_artifacts = false;
        } else {
            return 0;
        }
    }
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

static std::string issues_to_string(const std::vector<wb::issue> &issues) {
    std::ostringstream out;
    bool first = true;
    for (const wb::issue &issue : issues) {
        if (!first) out << "; ";
        first = false;
        out << issue.scope << ": " << issue.message;
    }
    return out.str();
}

static bool materialize_sliced_dataset_from_h5ad(const config &cfg,
                                                 const fs::path &h5ad_path,
                                                 const std::string &matrix_source,
                                                 const fs::path &output_path,
                                                 const fs::path &working_root,
                                                 double *materialize_ms,
                                                 wb::dataset_summary *summary_out,
                                                 wb::manifest_inspection *inspection_out,
                                                 std::string *error) {
    wb::source_entry source;
    wb::manifest_inspection inspection;
    wb::ingest_policy policy;
    wb::ingest_plan plan;
    wb::conversion_report report;

    source.dataset_id = h5ad_path.stem().string();
    source.matrix_path = h5ad_path.string();
    source.matrix_source = matrix_source;
    source.allow_processed = true;
    source.format = ::cellerator::ingest::dataset::source_h5ad;

    inspection = wb::inspect_source_entries({source}, "realPreprocessBench", cfg.reader_bytes);
    if (!inspection.ok || inspection.sources.empty()) {
        if (error != nullptr) *error = issues_to_string(inspection.issues);
        return false;
    }

    policy.max_part_nnz = cfg.max_part_nnz;
    policy.convert_window_bytes = cfg.convert_window_bytes;
    policy.target_shard_bytes = cfg.target_shard_bytes;
    policy.reader_bytes = cfg.reader_bytes;
    policy.output_path = output_path.string();
    policy.cache_dir.clear();
    policy.working_root = working_root.string();
    policy.verify_after_write = false;
    policy.device = cfg.device;
    policy.embed_metadata = true;
    policy.build_browse_cache = false;

    plan = wb::plan_dataset_ingest(inspection.sources, policy);
    if (!plan.ok) {
        if (error != nullptr) *error = issues_to_string(plan.issues);
        return false;
    }

    const auto begin = std::chrono::steady_clock::now();
    report = wb::convert_plan_to_dataset_csh5(plan);
    const auto end = std::chrono::steady_clock::now();
    if (!report.ok) {
        if (error != nullptr) *error = issues_to_string(report.issues);
        return false;
    }
    if (materialize_ms != nullptr) {
        *materialize_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    }
    if (inspection_out != nullptr) *inspection_out = inspection;
    if (summary_out != nullptr) {
        *summary_out = wb::summarize_dataset_csh5(output_path.string());
        if (!summary_out->ok) {
            if (error != nullptr) *error = issues_to_string(summary_out->issues);
            return false;
        }
    }
    return true;
}

static bool warm_sliced_dataset_cache(const fs::path &dataset_path,
                                      const std::string &cache_root,
                                      std::string *error) {
    if (cache_root.empty()) return true;
    if (cs::warm_dataset_sliced_ell_h5_cache(dataset_path.string().c_str(), cache_root.c_str())) return true;
    if (error != nullptr) *error = "failed to warm sliced cache";
    return false;
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
                                 double *python_ms,
                                 std::string *error) {
    const fs::path script_path = fs::path(__FILE__).parent_path() / "preprocess_scanpy_reference.py";
    const std::string cmd =
        shell_quote(cfg.python_exe) + " "
        + shell_quote(script_path.string())
        + " --h5ad " + shell_quote(cfg.h5ad_path)
        + " --matrix-source " + shell_quote(cfg.matrix_source)
        + " --blocked-analysis " + shell_quote(artifacts.analysis_blob.string())
        + " --sliced-analysis " + shell_quote(artifacts.analysis_blob.string())
        + " --summary-tsv " + shell_quote(artifacts.stability_summary_tsv.string())
        + " --details-json " + shell_quote(artifacts.stability_details_json.string())
        + " --target-sum " + std::to_string(cfg.target_sum)
        + " --min-counts " + std::to_string(cfg.min_counts)
        + " --min-genes " + std::to_string(cfg.min_genes)
        + " --max-mito-fraction " + std::to_string(cfg.max_mito_fraction)
        + " --min-gene-sum " + std::to_string(cfg.min_gene_sum)
        + " --min-detected-cells " + std::to_string(cfg.min_detected_cells)
        + " --min-variance " + std::to_string(cfg.min_variance);
    const auto begin = std::chrono::steady_clock::now();
    const int rc = std::system(cmd.c_str());
    const auto end = std::chrono::steady_clock::now();
    if (python_ms != nullptr) *python_ms = std::chrono::duration<double, std::milli>(end - begin).count();
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
    preprocess.finalize_after_preprocess = false;
    preprocess.cache_dir = cfg.cache_root;
    preprocess.working_root = cfg.working_root;
    return preprocess;
}

static double mean_ms(const std::vector<double> &values) {
    if (values.empty()) return 0.0;
    double total = 0.0;
    for (double value : values) total += value;
    return total / (double) values.size();
}

static void write_run_config(const config &cfg,
                             const dataset_artifacts &artifacts,
                             const wb::manifest_inspection &raw_inspection,
                             const wb::dataset_summary &filtered_output) {
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
    out << "  \"run_python_reference\": " << (cfg.run_python_reference ? "true" : "false") << ",\n";
    out << "  \"reuse_artifacts\": " << (cfg.reuse_artifacts ? "true" : "false") << ",\n";
    out << "  \"artifacts\": {\n";
    out << "    \"raw_sliced_input\": \"" << json_escape(artifacts.raw_sliced_input.string()) << "\",\n";
    out << "    \"analysis_blob\": \"" << json_escape(artifacts.analysis_blob.string()) << "\",\n";
    out << "    \"filtered_sliced_output\": \"" << json_escape(artifacts.filtered_sliced_output.string()) << "\"\n";
    out << "  },\n";
    out << "  \"raw_inspection\": {\n";
    out << "    \"rows\": " << (raw_inspection.sources.empty() ? 0ul : raw_inspection.sources.front().rows) << ",\n";
    out << "    \"cols\": " << (raw_inspection.sources.empty() ? 0ul : raw_inspection.sources.front().cols) << ",\n";
    out << "    \"nnz\": " << (raw_inspection.sources.empty() ? 0ul : raw_inspection.sources.front().nnz) << "\n";
    out << "  },\n";
    out << "  \"filtered_inspection\": {\n";
    out << "    \"rows\": " << filtered_output.rows << ",\n";
    out << "    \"cols\": " << filtered_output.cols << ",\n";
    out << "    \"nnz\": " << filtered_output.nnz << "\n";
    out << "  }\n";
    out << "}\n";
}

static void write_results(const dataset_artifacts &artifacts,
                          const config &cfg,
                          const wb::dataset_summary &raw_input,
                          const wb::dataset_summary &filtered_output,
                          const wb::preprocess_analysis_table &analysis,
                          const std::vector<double> &analyze_ms,
                          double materialize_raw_ms,
                          double repack_filtered_ms,
                          double python_ms,
                          double end_to_end_ms,
                          const std::map<std::string, std::string> &stability) {
    const std::uint64_t raw_input_file_bytes =
        fs::exists(artifacts.raw_sliced_input) ? (std::uint64_t) fs::file_size(artifacts.raw_sliced_input) : 0u;
    const std::uint64_t filtered_output_bytes =
        fs::exists(artifacts.filtered_sliced_output) ? (std::uint64_t) fs::file_size(artifacts.filtered_sliced_output) : 0u;
    std::ofstream out(artifacts.results_json, std::ios::trunc);
    out << "{\n";
    out << "  \"artifacts\": {\n";
    out << "    \"raw_sliced_input\": \"" << json_escape(artifacts.raw_sliced_input.string()) << "\",\n";
    out << "    \"filtered_sliced_output\": \"" << json_escape(artifacts.filtered_sliced_output.string()) << "\"\n";
    out << "  },\n";
    out << "  \"options\": {\n";
    out << "    \"run_python_reference\": " << (cfg.run_python_reference ? "true" : "false") << ",\n";
    out << "    \"use_all_devices\": " << (cfg.use_all_devices ? "true" : "false") << ",\n";
    out << "    \"reuse_artifacts\": " << (cfg.reuse_artifacts ? "true" : "false") << "\n";
    out << "  },\n";
    out << "  \"raw_input\": {\n";
    out << "    \"rows\": " << raw_input.rows << ",\n";
    out << "    \"cols\": " << raw_input.cols << ",\n";
    out << "    \"nnz\": " << raw_input.nnz << ",\n";
    out << "    \"partitions\": " << raw_input.num_partitions << ",\n";
    out << "    \"shards\": " << raw_input.num_shards << "\n";
    out << "  },\n";
    out << "  \"filtered_output\": {\n";
    out << "    \"rows\": " << filtered_output.rows << ",\n";
    out << "    \"cols\": " << filtered_output.cols << ",\n";
    out << "    \"nnz\": " << filtered_output.nnz << ",\n";
    out << "    \"partitions\": " << filtered_output.num_partitions << ",\n";
    out << "    \"shards\": " << filtered_output.num_shards << "\n";
    out << "  },\n";
    out << "  \"filter_counts\": {\n";
    out << "    \"kept_cells\": " << analysis.kept_cells << ",\n";
    out << "    \"kept_genes\": " << analysis.kept_genes << "\n";
    out << "  },\n";
    out << "  \"artifact_file_bytes\": {\n";
    out << "    \"raw_sliced_input\": " << raw_input_file_bytes << ",\n";
    out << "    \"filtered_sliced_output\": " << filtered_output_bytes << "\n";
    out << "  },\n";
    out << "  \"timings_ms\": {\n";
    out << "    \"materialize_raw_sliced\": " << materialize_raw_ms << ",\n";
    out << "    \"analyze_raw_sliced_mean\": " << mean_ms(analyze_ms) << ",\n";
    out << "    \"repack_filtered_sliced\": " << repack_filtered_ms << ",\n";
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
                          const wb::dataset_summary &raw_input,
                          const wb::dataset_summary &filtered_output,
                          const wb::preprocess_analysis_table &analysis,
                          const std::vector<double> &analyze_ms,
                          const std::map<std::string, std::string> &stability) {
    {
        std::ofstream out(artifacts.summary_json, std::ios::trunc);
        out << "{\n";
        out << "  \"raw_input_rows\": " << raw_input.rows << ",\n";
        out << "  \"filtered_output_rows\": " << filtered_output.rows << ",\n";
        out << "  \"raw_input_cols\": " << raw_input.cols << ",\n";
        out << "  \"filtered_output_cols\": " << filtered_output.cols << ",\n";
        out << "  \"analyze_raw_sliced_mean_ms\": " << mean_ms(analyze_ms) << ",\n";
        out << "  \"kept_cells\": " << analysis.kept_cells << ",\n";
        out << "  \"kept_genes\": " << analysis.kept_genes << ",\n";
        out << "  \"run_python_reference\": " << (cfg.run_python_reference ? "true" : "false") << ",\n";
        out << "  \"baseline_mode\": \"" << json_escape(stability.count("baseline_mode") != 0 ? stability.at("baseline_mode") : std::string()) << "\"\n";
        out << "}\n";
    }
    {
        std::ofstream out(artifacts.summary_txt, std::ios::trunc);
        out << "raw_sliced_input: " << raw_input.rows << " rows, " << raw_input.cols << " cols, " << raw_input.nnz << " nnz\n";
        out << "filtered_sliced_output: " << filtered_output.rows << " rows, " << filtered_output.cols << " cols, " << filtered_output.nnz << " nnz\n";
        out << "use_all_devices: " << (cfg.use_all_devices ? "true" : "false") << "\n";
        out << "analyze_raw_sliced_mean_ms: " << mean_ms(analyze_ms) << "\n";
        out << "kept_cells: " << analysis.kept_cells << "\n";
        out << "kept_genes: " << analysis.kept_genes << "\n";
        const auto it = stability.find("baseline_mode");
        out << "baseline_mode: " << (cfg.run_python_reference && it != stability.end() ? it->second : std::string("skipped")) << "\n";
        auto emit = [&](const char *key) {
            const auto found = stability.find(key);
            if (found != stability.end()) out << key << ": " << found->second << "\n";
        };
        emit("sliced.cell_keep.mismatch_count");
        emit("sliced.gene_keep.mismatch_count");
        emit("sliced.gene_sum.max_abs");
    }
}

int benchmark_main(int argc, char **argv) {
    config cfg;
    dataset_artifacts artifacts;
    wb::manifest_inspection raw_inspection;
    wb::dataset_summary raw_input_summary;
    wb::dataset_summary filtered_output_summary;
    wb::preprocess_analysis_table final_analysis;
    std::vector<double> analyze_raw_ms;
    std::map<std::string, std::string> stability_summary;
    double materialize_raw_ms = 0.0;
    double repack_filtered_ms = 0.0;
    double python_ms = 0.0;
    std::string error;
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

    artifacts.raw_sliced_input = fs::path(cfg.output_dir) / "raw.input.sliced.csh5";
    artifacts.filtered_sliced_output = fs::path(cfg.output_dir) / "filtered.output.sliced.csh5";
    artifacts.analysis_blob = fs::path(cfg.output_dir) / "analysis.sliced.bin";
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

    {
        scoped_nvtx_range range("materialize_raw_sliced");
        if (!cfg.reuse_artifacts || !fs::exists(artifacts.raw_sliced_input)) {
            if (!materialize_sliced_dataset_from_h5ad(cfg,
                                                      cfg.h5ad_path,
                                                      cfg.matrix_source,
                                                      artifacts.raw_sliced_input,
                                                      fs::path(cfg.working_root) / "raw",
                                                      &materialize_raw_ms,
                                                      &raw_input_summary,
                                                      &raw_inspection,
                                                      &error)) {
                std::fprintf(stderr, "realPreprocessBench: %s\n", error.c_str());
                return 1;
            }
        } else {
            raw_input_summary = wb::summarize_dataset_csh5(artifacts.raw_sliced_input.string());
            if (!raw_input_summary.ok) {
                std::fprintf(stderr, "realPreprocessBench: failed to summarize raw sliced input\n");
                return 1;
            }
        }
        if (!warm_sliced_dataset_cache(artifacts.raw_sliced_input, cfg.cache_root, &error)) {
            std::fprintf(stderr, "realPreprocessBench: %s for %s; continuing uncached\n",
                         error.c_str(),
                         artifacts.raw_sliced_input.string().c_str());
        }
    }

    for (unsigned int iter = 0u; iter < cfg.warmup + cfg.repeats; ++iter) {
        const bool measured = iter >= cfg.warmup;
        const unsigned int measured_index = iter - cfg.warmup;
        wb::preprocess_config analyze_cfg = make_preprocess_config(cfg);
        wb::preprocess_analysis_table analysis;
        scoped_nvtx_range range("steady_state_compute.sliced");
        const auto phase_begin = std::chrono::steady_clock::now();
        analysis = wb::analyze_dataset_preprocess(artifacts.raw_sliced_input.string(), analyze_cfg);
        const auto phase_end = std::chrono::steady_clock::now();
        if (!analysis.ok) {
            cs::sharded<cs::sparse::sliced_ell> probe_matrix;
            cs::shard_storage probe_storage;
            const int header_only_ok = ([](const std::string &dataset_path,
                                           cs::sharded<cs::sparse::sliced_ell> *matrix) -> int {
                cs::init(matrix);
                return cs::load_dataset_sliced_ell_h5_header(dataset_path.c_str(), matrix, nullptr);
            })(artifacts.raw_sliced_input.string(), &probe_matrix);
            cs::init(&probe_storage);
            const int header_with_storage_ok =
                cs::load_dataset_sliced_ell_h5_header(artifacts.raw_sliced_input.string().c_str(), &probe_matrix, &probe_storage);
            std::fprintf(stderr, "realPreprocessBench: sliced analyze failed\n");
            std::fprintf(stderr,
                         "  header_only_ok=%d header_with_storage_ok=%d\n",
                         header_only_ok,
                         header_with_storage_ok);
            for (const wb::issue &issue : analysis.issues) {
                std::fprintf(stderr, "  [%s] %s: %s\n",
                             wb::severity_name(issue.severity).c_str(),
                             issue.scope.c_str(),
                             issue.message.c_str());
            }
            cs::clear(&probe_storage);
            cs::clear(&probe_matrix);
            return 1;
        }
        if (measured) analyze_raw_ms.push_back(std::chrono::duration<double, std::milli>(phase_end - phase_begin).count());
        if (measured && measured_index + 1u == cfg.repeats) final_analysis = std::move(analysis);
    }

    if (!write_analysis_blob(artifacts.analysis_blob, final_analysis, &error)) {
        std::fprintf(stderr, "realPreprocessBench: %s\n", error.c_str());
        return 1;
    }

    {
        scoped_nvtx_range range("repack_filtered_sliced");
        if (!cfg.reuse_artifacts || !fs::exists(artifacts.filtered_sliced_output)) {
            wb::preprocess_config finalize_cfg = make_preprocess_config(cfg);
            finalize_cfg.finalize_after_preprocess = true;
            const auto phase_begin = std::chrono::steady_clock::now();
            wb::preprocess_persist_summary finalize_summary =
                wb::persist_preprocess_analysis_to_output(artifacts.raw_sliced_input.string(),
                                                          artifacts.filtered_sliced_output.string(),
                                                          final_analysis,
                                                          finalize_cfg);
            const auto phase_end = std::chrono::steady_clock::now();
            repack_filtered_ms = std::chrono::duration<double, std::milli>(phase_end - phase_begin).count();
            if (!finalize_summary.summary.ok) {
                std::fprintf(stderr, "realPreprocessBench: sliced finalize failed\n");
                for (const wb::issue &issue : finalize_summary.summary.issues) {
                    std::fprintf(stderr, "  [%s] %s: %s\n",
                                 wb::severity_name(issue.severity).c_str(),
                                 issue.scope.c_str(),
                                 issue.message.c_str());
                }
                return 1;
            }
            filtered_output_summary = wb::summarize_dataset_csh5(artifacts.filtered_sliced_output.string());
            if (!filtered_output_summary.ok) {
                std::fprintf(stderr, "realPreprocessBench: failed to summarize filtered sliced output\n");
                return 1;
            }
        } else {
            filtered_output_summary = wb::summarize_dataset_csh5(artifacts.filtered_sliced_output.string());
            if (!filtered_output_summary.ok) {
                std::fprintf(stderr, "realPreprocessBench: failed to summarize filtered sliced output\n");
                return 1;
            }
        }
        if (!warm_sliced_dataset_cache(artifacts.filtered_sliced_output, cfg.cache_root, &error)) {
            std::fprintf(stderr, "realPreprocessBench: %s for %s; continuing uncached\n",
                         error.c_str(),
                         artifacts.filtered_sliced_output.string().c_str());
        }
    }

    if (cfg.run_python_reference) {
        scoped_nvtx_range range("python_reference");
        if (!run_python_reference(cfg, artifacts, &stability_summary, &python_ms, &error)) {
            std::fprintf(stderr, "realPreprocessBench: %s\n", error.c_str());
            return 1;
        }
    } else {
        stability_summary["baseline_mode"] = "skipped";
    }

    write_run_config(cfg, artifacts, raw_inspection, filtered_output_summary);

    const auto end_to_end_end = std::chrono::steady_clock::now();
    const double end_to_end_ms = std::chrono::duration<double, std::milli>(end_to_end_end - end_to_end_begin).count();

    write_results(artifacts,
                  cfg,
                  raw_input_summary,
                  filtered_output_summary,
                  final_analysis,
                  analyze_raw_ms,
                  materialize_raw_ms,
                  repack_filtered_ms,
                  python_ms,
                  end_to_end_ms,
                  stability_summary);
    write_summary(artifacts,
                  cfg,
                  raw_input_summary,
                  filtered_output_summary,
                  final_analysis,
                  analyze_raw_ms,
                  stability_summary);

    std::printf("raw_analyze_mean_ms=%.3f kept_cells=%.0f kept_genes=%lu repack_ms=%.3f baseline=%s\n",
                mean_ms(analyze_raw_ms),
                final_analysis.kept_cells,
                final_analysis.kept_genes,
                repack_filtered_ms,
                stability_summary.count("baseline_mode") != 0 ? stability_summary["baseline_mode"].c_str() : "unknown");
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    return benchmark_main(argc, argv);
}
