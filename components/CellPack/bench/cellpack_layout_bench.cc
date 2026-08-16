#include <CellPack/layout_selector.hh>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "benchmark_mutex.hh"

namespace {

using clock_type = std::chrono::steady_clock;

struct bench_options {
    std::string scenario = "regular_modules";
    std::string output_dir;
    cellpack::u32 rows = 4096u;
    cellpack::u32 features = 2048u;
    cellpack::u32 modules = 32u;
    int warmup = 1;
    int repeats = 5;
};

struct synthetic_fixture {
    std::vector<cellpack::u32> feature_modules;
    std::vector<cellpack::u32> signature_offsets;
    std::vector<cellpack::u32> signature_modules;
    std::vector<cellpack::u32> row_offsets;
    std::vector<cellpack::u32> feature_ids;
    std::vector<float> values;
};

struct phase_times {
    double generate_ms = 0.0;
    double plan_ms = 0.0;
    double select_layout_ms = 0.0;
    double estimate_runtime_ms = 0.0;
    double baseline_reference_ms = 0.0;
};

struct bench_result {
    bench_options options;
    phase_times phases;
    cellpack::layout_plan_summary source_summary;
    cellpack::layout_selection_plan selection;
    cellpack::u64 csr_baseline_bytes = 0u;
    cellpack::u64 blocked_baseline_bytes = 0u;
};

double elapsed_ms(clock_type::time_point begin, clock_type::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

cellpack::u32 parse_u32(const char *text, const char *name) {
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value > 0xfffffffful) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + text);
    }
    return static_cast<cellpack::u32>(value);
}

int parse_int(const char *text, const char *name) {
    char *end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 || value > 1000000) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + text);
    }
    return static_cast<int>(value);
}

bench_options parse_args(int argc, char **argv) {
    bench_options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char *name) -> const char * {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (arg == "--scenario") options.scenario = require_value("--scenario");
        else if (arg == "--rows") options.rows = parse_u32(require_value("--rows"), "rows");
        else if (arg == "--features") options.features = parse_u32(require_value("--features"), "features");
        else if (arg == "--modules") options.modules = parse_u32(require_value("--modules"), "modules");
        else if (arg == "--warmup") options.warmup = parse_int(require_value("--warmup"), "warmup");
        else if (arg == "--repeats") options.repeats = parse_int(require_value("--repeats"), "repeats");
        else if (arg == "--output-dir") options.output_dir = require_value("--output-dir");
        else throw std::runtime_error("unknown argument: " + arg);
    }
    if (options.rows == 0u || options.features == 0u || options.modules == 0u) {
        throw std::runtime_error("rows, features, and modules must be nonzero");
    }
    if (options.repeats <= 0) {
        throw std::runtime_error("repeats must be nonzero");
    }
    options.modules = std::min(options.modules, options.features);
    return options;
}

void append_unique_sorted(std::vector<cellpack::u32> *features, cellpack::u32 value) {
    features->push_back(value);
}

std::vector<cellpack::u32> make_row_features(
    const bench_options &options,
    cellpack::u32 row,
    cellpack::u32 module_width) {
    std::vector<cellpack::u32> features;
    const cellpack::u32 module = row % options.modules;
    const cellpack::u32 begin = module * module_width;
    const cellpack::u32 end = std::min(options.features, begin + module_width);
    cellpack::u32 active = 0u;
    if (options.scenario == "regular_modules") active = std::min<cellpack::u32>(module_width, 24u);
    else if (options.scenario == "skewed_rows") active = (row % 17u == 0u) ? std::min<cellpack::u32>(module_width, 96u) : std::min<cellpack::u32>(module_width, 8u);
    else if (options.scenario == "high_residual_fraction") active = std::min<cellpack::u32>(module_width, 10u);
    else if (options.scenario == "dense_tiles") active = std::min<cellpack::u32>(module_width, (module_width * 7u) / 8u);
    else if (options.scenario == "adversarial_low_fill") active = std::min<cellpack::u32>(module_width, 2u);
    else throw std::runtime_error("unknown scenario: " + options.scenario);

    for (cellpack::u32 i = 0; i < active && begin + i < end; ++i) {
        append_unique_sorted(&features, begin + i);
    }
    if (options.scenario == "high_residual_fraction" && options.features > options.modules * module_width) {
        for (cellpack::u32 i = 0; i < 6u; ++i) {
            const cellpack::u32 residual_feature = options.modules * module_width
                + ((row * 17u + i * 31u) % (options.features - options.modules * module_width));
            append_unique_sorted(&features, residual_feature);
        }
    }
    if (options.scenario == "adversarial_low_fill" && options.features > 16u) {
        append_unique_sorted(&features, (row * 131u + 7u) % options.features);
    }
    std::sort(features.begin(), features.end());
    features.erase(std::unique(features.begin(), features.end()), features.end());
    return features;
}

synthetic_fixture generate_fixture(const bench_options &options) {
    synthetic_fixture fixture;
    const cellpack::u32 module_width = std::max<cellpack::u32>(1u, options.features / options.modules);
    fixture.feature_modules.resize(options.features, cellpack::default_residual_module_id);
    for (cellpack::u32 module = 0; module < options.modules; ++module) {
        const cellpack::u32 begin = module * module_width;
        const cellpack::u32 end = std::min(options.features, begin + module_width);
        for (cellpack::u32 feature = begin; feature < end; ++feature) {
            fixture.feature_modules[feature] = module;
        }
    }

    fixture.signature_offsets.reserve(static_cast<std::size_t>(options.rows) + 1u);
    fixture.row_offsets.reserve(static_cast<std::size_t>(options.rows) + 1u);
    fixture.signature_offsets.push_back(0u);
    fixture.row_offsets.push_back(0u);
    for (cellpack::u32 row = 0; row < options.rows; ++row) {
        const cellpack::u32 module = row % options.modules;
        fixture.signature_modules.push_back(module);
        if (options.scenario == "high_residual_fraction") {
            fixture.signature_modules.push_back(cellpack::default_residual_module_id);
        }
        fixture.signature_offsets.push_back(static_cast<cellpack::u32>(fixture.signature_modules.size()));

        std::vector<cellpack::u32> row_features = make_row_features(options, row, module_width);
        fixture.feature_ids.insert(fixture.feature_ids.end(), row_features.begin(), row_features.end());
        fixture.values.insert(fixture.values.end(), row_features.size(), 1.0f);
        fixture.row_offsets.push_back(static_cast<cellpack::u32>(fixture.feature_ids.size()));
    }
    return fixture;
}

bench_result run_once(const bench_options &options) {
    bench_result result;
    result.options = options;

    clock_type::time_point begin = clock_type::now();
    synthetic_fixture fixture = generate_fixture(options);
    clock_type::time_point after_generate = clock_type::now();

    cellpack::feature_module_assignment_view features;
    features.feature_to_module = fixture.feature_modules.data();
    features.feature_count = options.features;
    features.residual_module_id = cellpack::default_residual_module_id;
    cellpack::row_signature_view rows;
    rows.row_count = options.rows;
    rows.row_offsets = fixture.signature_offsets.data();
    rows.module_ids = fixture.signature_modules.data();
    rows.entry_count = static_cast<cellpack::u32>(fixture.signature_modules.size());
    cellpack::planner_config planner_config;
    planner_config.residual_module_id = cellpack::default_residual_module_id;
    planner_config.min_primary_rows = 2u;

    cellpack::static_plan plan;
    cellpack::validation_result validation = cellpack::build_static_plan(features, rows, planner_config, &plan);
    require(static_cast<bool>(validation), validation.message);

    cellpack::csr_view csr;
    csr.row_count = options.rows;
    csr.feature_count = options.features;
    csr.nnz_count = static_cast<cellpack::u32>(fixture.feature_ids.size());
    csr.row_offsets = fixture.row_offsets.data();
    csr.feature_ids = fixture.feature_ids.data();
    csr.values = fixture.values.data();
    cellpack::packed_coordinate_plan packed;
    validation = cellpack::build_packed_coordinate_plan(csr, plan, &packed);
    require(static_cast<bool>(validation), validation.message);
    clock_type::time_point after_plan = clock_type::now();

    cellpack::layout_metrics_plan metrics;
    validation = cellpack::build_layout_metrics(plan, packed, cellpack::layout_metrics_config{}, &metrics);
    require(static_cast<bool>(validation), validation.message);
    result.source_summary = cellpack::summarize_layout_metrics(metrics);
    cellpack::layout_selector_config selector_config;
    validation = cellpack::select_layouts(metrics, selector_config, &result.selection);
    require(static_cast<bool>(validation), validation.message);
    clock_type::time_point after_select = clock_type::now();

    for (const cellpack::region_layout_metrics &region : metrics.regions) {
        result.csr_baseline_bytes += cellpack::csr_estimated_bytes(region);
        result.blocked_baseline_bytes += cellpack::blocked_ell_estimated_bytes(region);
    }
    clock_type::time_point after_estimate = clock_type::now();
    volatile cellpack::u64 baseline_sink = result.csr_baseline_bytes + result.blocked_baseline_bytes;
    (void) baseline_sink;
    clock_type::time_point after_baseline = clock_type::now();

    result.phases.generate_ms = elapsed_ms(begin, after_generate);
    result.phases.plan_ms = elapsed_ms(after_generate, after_plan);
    result.phases.select_layout_ms = elapsed_ms(after_plan, after_select);
    result.phases.estimate_runtime_ms = elapsed_ms(after_select, after_estimate);
    result.phases.baseline_reference_ms = elapsed_ms(after_estimate, after_baseline);
    return result;
}

void write_summary_text(const bench_result &result, std::ostream &out) {
    out << "scenario: " << result.options.scenario << "\n";
    out << "rows: " << result.options.rows << "\n";
    out << "features: " << result.options.features << "\n";
    out << "modules: " << result.options.modules << "\n";
    out << "equivalence: structural_coverage\n";
    out << "phases_ms.generate: " << result.phases.generate_ms << "\n";
    out << "phases_ms.plan: " << result.phases.plan_ms << "\n";
    out << "phases_ms.select_layout: " << result.phases.select_layout_ms << "\n";
    out << "phases_ms.estimate_runtime: " << result.phases.estimate_runtime_ms << "\n";
    out << "phases_ms.baseline_reference: " << result.phases.baseline_reference_ms << "\n";
    out << "impl_a_csr_estimated_bytes: " << result.csr_baseline_bytes << "\n";
    out << "impl_b_blocked_ell_estimated_bytes: " << result.blocked_baseline_bytes << "\n";
    out << "impl_c_hybrid_estimated_bytes: " << result.selection.summary.total_estimated_bytes << "\n";
    out << "hybrid_launch_groups: " << result.selection.summary.launch_group_count << "\n";
    out << "hybrid_residual_nnz_fraction: " << result.selection.summary.residual_nnz_fraction << "\n";
    out << "hybrid_dense_tile_coverage: " << result.selection.summary.dense_tile_candidate_coverage << "\n";
}

void write_summary_json(const bench_result &result, std::ostream &out) {
    out << "{\n";
    out << "  \"compare_config\": {\n";
    out << "    \"comparison_id\": \"cellpack-m3-layout-selection\",\n";
    out << "    \"impl_a_name\": \"csr_reference_estimate\",\n";
    out << "    \"impl_b_name\": \"blocked_ell_reference_estimate\",\n";
    out << "    \"impl_c_name\": \"cellpack_hybrid_selection\",\n";
    out << "    \"scenario_id\": \"" << result.options.scenario << "\",\n";
    out << "    \"warmup\": " << result.options.warmup << ",\n";
    out << "    \"repeats\": " << result.options.repeats << ",\n";
    out << "    \"mutex_path\": \"CUDA_V100_BENCHMARK_MUTEX_PATH or /tmp/cuda_v100_benchmark.lock\"\n";
    out << "  },\n";
    out << "  \"status\": \"ok\",\n";
    out << "  \"equivalence\": \"structural_coverage\",\n";
    out << "  \"phases_ms\": {\n";
    out << "    \"generate\": " << result.phases.generate_ms << ",\n";
    out << "    \"plan\": " << result.phases.plan_ms << ",\n";
    out << "    \"select_layout\": " << result.phases.select_layout_ms << ",\n";
    out << "    \"estimate_runtime\": " << result.phases.estimate_runtime_ms << ",\n";
    out << "    \"baseline_reference\": " << result.phases.baseline_reference_ms << "\n";
    out << "  },\n";
    out << "  \"metrics\": {\n";
    out << "    \"csr_estimated_bytes\": " << result.csr_baseline_bytes << ",\n";
    out << "    \"blocked_ell_estimated_bytes\": " << result.blocked_baseline_bytes << ",\n";
    out << "    \"hybrid_estimated_bytes\": " << result.selection.summary.total_estimated_bytes << ",\n";
    out << "    \"hybrid_launch_groups\": " << result.selection.summary.launch_group_count << ",\n";
    out << "    \"hybrid_residual_nnz_fraction\": " << result.selection.summary.residual_nnz_fraction << ",\n";
    out << "    \"hybrid_dense_tile_coverage\": " << result.selection.summary.dense_tile_candidate_coverage << "\n";
    out << "  }\n";
    out << "}\n";
}

void write_outputs(const bench_result &result) {
    write_summary_text(result, std::cout);
    if (result.options.output_dir.empty()) return;
    const std::string text_path = result.options.output_dir + "/summary.txt";
    const std::string json_path = result.options.output_dir + "/summary.json";
    std::ofstream text_out(text_path);
    std::ofstream json_out(json_path);
    if (!text_out || !json_out) {
        throw std::runtime_error("failed to open benchmark summary output files");
    }
    write_summary_text(result, text_out);
    write_summary_json(result, json_out);
}

} // namespace

int main(int argc, char **argv) {
    try {
        const bench_options options = parse_args(argc, argv);
        cellerator::bench::benchmark_mutex_guard benchmark_mutex("cellPackLayoutBench");
        bench_result final_result;
        for (int i = 0; i < options.warmup; ++i) {
            (void) run_once(options);
        }
        for (int i = 0; i < options.repeats; ++i) {
            final_result = run_once(options);
        }
        write_outputs(final_result);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "cellPackLayoutBench: %s\n", error.what());
        return 1;
    }
    return 0;
}
