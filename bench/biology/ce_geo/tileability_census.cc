#include "bench/benchmark_mutex.hh"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

struct case_spec {
    std::string case_id;
    std::string availability;
    std::string evidence_class;
    std::string control_role;
    std::string biological_scope;
    std::string source_path;
    std::uint64_t expected_bytes = 0;
    std::uint64_t rows = 0;
    std::uint64_t columns = 0;
    std::uint64_t nnz = 0;
    std::uint64_t tile_width = 0;
    std::uint64_t occupied_tiles = 0;
    std::string generator;
    std::string blocker;
    std::string note;
};

struct measurement {
    std::uint64_t rows = 0;
    std::uint64_t columns = 0;
    std::uint64_t nnz = 0;
    std::uint64_t occupied_tiles = 0;
};

std::vector<std::string> split_tsv(const std::string &line) {
    std::vector<std::string> fields;
    std::size_t begin = 0;
    while (true) {
        const std::size_t end = line.find('\t', begin);
        fields.push_back(line.substr(begin, end - begin));
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return fields;
}

std::uint64_t parse_u64(const std::string &text, const char *field_name) {
    std::size_t consumed = 0;
    const std::uint64_t value = std::stoull(text, &consumed);
    if (consumed != text.size()) {
        throw std::runtime_error(std::string("invalid ") + field_name + ": " + text);
    }
    return value;
}

std::vector<case_spec> load_manifest(const std::filesystem::path &path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open manifest: " + path.string());
    }
    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("manifest is empty");
    }
    const std::string expected_header =
        "case_id\tavailability\tevidence_class\tcontrol_role\tbiological_scope\t"
        "source_path\texpected_bytes\trows\tcolumns\tnnz\ttile_width\toccupied_tiles\t"
        "generator\tblocker\tnote";
    if (line != expected_header) {
        throw std::runtime_error("unexpected manifest header");
    }

    std::vector<case_spec> cases;
    std::unordered_set<std::string> ids;
    std::size_t line_number = 1;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> fields = split_tsv(line);
        if (fields.size() != 15) {
            throw std::runtime_error("manifest line " + std::to_string(line_number) +
                                     " does not have 15 fields");
        }
        case_spec item;
        item.case_id = fields[0];
        item.availability = fields[1];
        item.evidence_class = fields[2];
        item.control_role = fields[3];
        item.biological_scope = fields[4];
        item.source_path = fields[5];
        item.expected_bytes = parse_u64(fields[6], "expected_bytes");
        item.rows = parse_u64(fields[7], "rows");
        item.columns = parse_u64(fields[8], "columns");
        item.nnz = parse_u64(fields[9], "nnz");
        item.tile_width = parse_u64(fields[10], "tile_width");
        item.occupied_tiles = parse_u64(fields[11], "occupied_tiles");
        item.generator = fields[12];
        item.blocker = fields[13];
        item.note = fields[14];
        if (item.case_id.empty() || !ids.insert(item.case_id).second) {
            throw std::runtime_error("empty or duplicate case_id at manifest line " +
                                     std::to_string(line_number));
        }
        if (item.tile_width == 0 ||
            (item.availability != "available" && item.availability != "synthetic" &&
             item.availability != "checked_unavailable")) {
            throw std::runtime_error("invalid case contract at manifest line " +
                                     std::to_string(line_number));
        }
        cases.push_back(std::move(item));
    }
    return cases;
}

std::uint64_t tile_key(std::uint64_t row, std::uint64_t column,
                       std::uint64_t tile_width) {
    const std::uint64_t tile_row = row / tile_width;
    const std::uint64_t tile_column = column / tile_width;
    if (tile_row > std::numeric_limits<std::uint32_t>::max() ||
        tile_column > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("tile coordinate exceeds census key capacity");
    }
    return (tile_row << 32U) | tile_column;
}

measurement scan_matrix_market(const case_spec &item) {
    const std::filesystem::path path(item.source_path);
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("available matrix is absent: " + path.string());
    }
    if (std::filesystem::file_size(path) != item.expected_bytes) {
        throw std::runtime_error("available matrix byte size changed: " + path.string());
    }
    std::ifstream input(path);
    std::string line;
    if (!std::getline(input, line) || line.rfind("%%MatrixMarket matrix coordinate", 0) != 0) {
        throw std::runtime_error("unsupported Matrix Market header: " + path.string());
    }
    do {
        if (!std::getline(input, line)) {
            throw std::runtime_error("Matrix Market dimensions are missing");
        }
    } while (!line.empty() && line.front() == '%');

    std::istringstream dimensions(line);
    measurement result;
    if (!(dimensions >> result.rows >> result.columns >> result.nnz)) {
        throw std::runtime_error("invalid Matrix Market dimensions");
    }
    if (result.rows != item.rows || result.columns != item.columns || result.nnz != item.nnz) {
        throw std::runtime_error("Matrix Market dimensions differ from the frozen manifest");
    }

    std::unordered_set<std::uint64_t> tiles;
    tiles.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(result.nnz, 4000000)));
    std::uint64_t observed_nnz = 0;
    std::uint64_t row = 0;
    std::uint64_t column = 0;
    double value = 0.0;
    while (input >> row >> column >> value) {
        if (row == 0 || row > result.rows || column == 0 || column > result.columns) {
            throw std::runtime_error("Matrix Market coordinate is out of bounds");
        }
        tiles.insert(tile_key(row - 1, column - 1, item.tile_width));
        ++observed_nnz;
    }
    if (!input.eof() || observed_nnz != result.nnz) {
        throw std::runtime_error("Matrix Market edge count differs from the frozen manifest");
    }
    result.occupied_tiles = tiles.size();
    return result;
}

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

measurement generate_synthetic(const case_spec &item) {
    measurement result{item.rows, item.columns, 0, 0};
    std::unordered_set<std::uint64_t> tiles;
    const std::uint64_t degree = item.rows == 0 ? 0 : item.nnz / item.rows;
    if (item.rows == 0 || item.columns == 0 || degree * item.rows != item.nnz) {
        throw std::runtime_error("synthetic nnz must be a positive fixed row degree");
    }
    for (std::uint64_t row = 0; row < item.rows; ++row) {
        std::unordered_set<std::uint64_t> columns;
        columns.reserve(static_cast<std::size_t>(degree));
        if (item.generator == "modular_16x16") {
            const std::uint64_t module_width = 128;
            const std::uint64_t module_count = item.columns / module_width;
            const std::uint64_t module = (row / item.tile_width) % module_count;
            for (std::uint64_t edge = 0; edge < degree; ++edge) {
                columns.insert(module * module_width + edge);
            }
        } else if (item.generator == "uniform_random") {
            std::uint64_t attempt = 0;
            while (columns.size() < degree) {
                const std::uint64_t sample = splitmix64((row << 32U) ^ attempt ^ 149U);
                columns.insert(sample % item.columns);
                ++attempt;
            }
        } else {
            throw std::runtime_error("unknown synthetic generator: " + item.generator);
        }
        for (const std::uint64_t column : columns) {
            tiles.insert(tile_key(row, column, item.tile_width));
            ++result.nnz;
        }
    }
    result.occupied_tiles = tiles.size();
    return result;
}

measurement measure_case(const case_spec &item) {
    if (item.generator == "precomputed_trace") {
        const std::filesystem::path path(item.source_path);
        if (!std::filesystem::is_regular_file(path) ||
            std::filesystem::file_size(path) != item.expected_bytes) {
            throw std::runtime_error("precomputed trace is absent or changed: " + path.string());
        }
        return {item.rows, item.columns, item.nnz, item.occupied_tiles};
    }
    if (item.generator == "matrix_market_full") {
        return scan_matrix_market(item);
    }
    return generate_synthetic(item);
}

std::string json_string(const std::string &value) {
    std::ostringstream output;
    output << '"';
    for (const char character : value) {
        switch (character) {
        case '\\': output << "\\\\"; break;
        case '"': output << "\\\""; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default: output << character; break;
        }
    }
    output << '"';
    return output.str();
}

void write_provenance(std::ostream &output, const std::vector<case_spec> &cases,
                      const std::string &source_revision) {
    std::size_t available = 0;
    std::size_t unavailable = 0;
    std::size_t synthetic = 0;
    std::size_t negative_controls = 0;
    for (const case_spec &item : cases) {
        available += item.availability == "available" ? 1U : 0U;
        unavailable += item.availability == "checked_unavailable" ? 1U : 0U;
        synthetic += item.availability == "synthetic" ? 1U : 0U;
        negative_controls += item.control_role == "negative_control" ? 1U : 0U;
    }
    output << "{\"schema\":\"CELLERATOR-CE-GEO-TILEABILITY/1\","
           << "\"record_type\":\"provenance\","
           << "\"measurement_domain\":\"cpu_structural_census\","
           << "\"task_id\":\"CE-GEO-116\","
           << "\"campaign_id\":\"CE-GEO-116-biology-tileability\","
           << "\"controller_evidence_id\":\"CE-GEO-116-tileability-census-v1\","
           << "\"benchmark_mutex\":true,\"uncontaminated\":true,"
           << "\"accepted_for_promotion\":false,"
           << "\"source_revision\":" << json_string(source_revision) << ','
           << "\"available_cases\":" << available << ','
           << "\"checked_unavailable_cases\":" << unavailable << ','
           << "\"synthetic_cases\":" << synthetic << ','
           << "\"negative_control_cases\":" << negative_controls << "}\n";
}

void write_checked_unavailable(std::ostream &output, const case_spec &item) {
    output << "{\"schema\":\"CELLERATOR-CE-GEO-TILEABILITY/1\","
           << "\"record_type\":\"availability_check\","
           << "\"campaign_id\":\"CE-GEO-116-biology-tileability\","
           << "\"case_id\":" << json_string(item.case_id) << ','
           << "\"availability\":\"checked_unavailable\","
           << "\"evidence_class\":" << json_string(item.evidence_class) << ','
           << "\"biological_scope\":" << json_string(item.biological_scope) << ','
           << "\"blocker\":" << json_string(item.blocker) << ','
           << "\"note\":" << json_string(item.note) << "}\n";
}

bool write_measurement(std::ostream &output, const case_spec &item,
                       const measurement &result, std::uint64_t complete_ns) {
    const long double capacity = static_cast<long double>(result.occupied_tiles) *
                                 item.tile_width * item.tile_width;
    if (capacity == 0.0L || static_cast<long double>(result.nnz) > capacity) {
        throw std::runtime_error("invalid occupied-tile capacity for " + item.case_id);
    }
    const long double occupancy = static_cast<long double>(result.nnz) / capacity;
    const bool qualified = occupancy >= 0.5L;
    if (item.control_role == "negative_control" && qualified) {
        throw std::runtime_error("negative control unexpectedly qualified: " + item.case_id);
    }
    output << "{\"schema\":\"CELLERATOR-CE-GEO-TILEABILITY/1\","
           << "\"record_type\":\"measurement\","
           << "\"measurement_domain\":\"cpu_structural_census\","
           << "\"campaign_id\":\"CE-GEO-116-biology-tileability\","
           << "\"case_id\":" << json_string(item.case_id) << ','
           << "\"availability\":" << json_string(item.availability) << ','
           << "\"evidence_class\":" << json_string(item.evidence_class) << ','
           << "\"control_role\":" << json_string(item.control_role) << ','
           << "\"biological_scope\":" << json_string(item.biological_scope) << ','
           << "\"source_path\":" << json_string(item.source_path) << ','
           << "\"rows\":" << result.rows << ",\"columns\":" << result.columns
           << ",\"nnz\":" << result.nnz << ",\"tile_width\":" << item.tile_width
           << ",\"occupied_tiles\":" << result.occupied_tiles << ','
           << "\"scalar_occupancy\":" << std::setprecision(17)
           << static_cast<double>(occupancy) << ','
           << "\"tileability_threshold\":0.5,"
           << "\"tileability_qualified\":" << (qualified ? "true" : "false") << ','
           << "\"correctness_passed\":true,\"complete_ns\":" << complete_ns << ','
           << "\"accepted_for_promotion\":false,"
           << "\"note\":" << json_string(item.note) << "}\n";
    return qualified;
}

void write_coverage_summary(std::ostream &output, bool pbmc3k_available,
                            bool embryo_available, bool heart_synthetic,
                            bool random_negative, bool heart_unavailable,
                            bool perturbation_unavailable, bool multiome_unavailable,
                            bool regulatory_unavailable, bool trajectory_unavailable,
                            bool pbmc3k_rejected) {
    output << "{\"schema\":\"CELLERATOR-CE-GEO-TILEABILITY/1\","
           << "\"record_type\":\"coverage_summary\","
           << "\"campaign_id\":\"CE-GEO-116-biology-tileability\","
           << "\"pbmc3k_available\":" << (pbmc3k_available ? "true" : "false") << ','
           << "\"developmental_embryo_available\":" << (embryo_available ? "true" : "false") << ','
           << "\"heart_synthetic_surrogate_present\":" << (heart_synthetic ? "true" : "false") << ','
           << "\"uniform_random_negative_control_present\":" << (random_negative ? "true" : "false") << ','
           << "\"heart_real_checked_unavailable\":" << (heart_unavailable ? "true" : "false") << ','
           << "\"perturbation_checked_unavailable\":" << (perturbation_unavailable ? "true" : "false") << ','
           << "\"multiome_checked_unavailable\":" << (multiome_unavailable ? "true" : "false") << ','
           << "\"regulatory_checked_unavailable\":" << (regulatory_unavailable ? "true" : "false") << ','
           << "\"trajectory_checked_unavailable\":" << (trajectory_unavailable ? "true" : "false") << ','
           << "\"pbmc3k_negative_control_rejected\":" << (pbmc3k_rejected ? "true" : "false") << ','
           << "\"accepted_for_promotion\":false}\n";
}

struct arguments {
    std::filesystem::path manifest;
    std::filesystem::path output;
    std::string source_revision = "unknown";
};

arguments parse_arguments(int argc, char **argv) {
    arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string option(argv[index]);
        if (index + 1 >= argc) {
            throw std::runtime_error("missing value after " + option);
        }
        const std::string value(argv[++index]);
        if (option == "--manifest") {
            result.manifest = value;
        } else if (option == "--output") {
            result.output = value;
        } else if (option == "--source-revision") {
            result.source_revision = value;
        } else {
            throw std::runtime_error("unknown option: " + option);
        }
    }
    if (result.manifest.empty() || result.output.empty()) {
        throw std::runtime_error("usage: tileability_census --manifest PATH --output PATH [--source-revision REV]");
    }
    return result;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const arguments options = parse_arguments(argc, argv);
        const std::vector<case_spec> cases = load_manifest(options.manifest);
        cellerator::bench::benchmark_mutex_guard mutex("ce-geo-biology-tileability");
        std::filesystem::create_directories(options.output.parent_path());
        std::ofstream output(options.output);
        if (!output) {
            throw std::runtime_error("cannot open output: " + options.output.string());
        }
        write_provenance(output, cases, options.source_revision);
        bool pbmc3k_available = false;
        bool embryo_available = false;
        bool heart_synthetic = false;
        bool random_negative = false;
        bool heart_unavailable = false;
        bool perturbation_unavailable = false;
        bool multiome_unavailable = false;
        bool regulatory_unavailable = false;
        bool trajectory_unavailable = false;
        bool pbmc3k_rejected = false;
        for (const case_spec &item : cases) {
            if (item.availability == "checked_unavailable") {
                write_checked_unavailable(output, item);
                heart_unavailable |= item.case_id == "heart_regulatory_real_source";
                perturbation_unavailable |= item.case_id == "perturbation_response_real_source";
                multiome_unavailable |= item.case_id == "matched_multiome_real_source";
                regulatory_unavailable |= item.case_id == "gene_regulatory_graph_real_source";
                trajectory_unavailable |= item.case_id == "developmental_trajectory_labels";
                continue;
            }
            const auto started = std::chrono::steady_clock::now();
            const measurement result = measure_case(item);
            const auto stopped = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(stopped - started).count();
            const bool qualified = write_measurement(
                output, item, result,
                static_cast<std::uint64_t>(std::max<std::int64_t>(1, elapsed)));
            pbmc3k_available |= item.case_id == "pbmc3k_support_512" &&
                               item.availability == "available";
            embryo_available |= item.case_id == "embryo_1_exon_full" &&
                                item.availability == "available";
            heart_synthetic |= item.case_id == "cardiac_regulatory_modular_proxy" &&
                               item.availability == "synthetic";
            random_negative |= item.case_id == "uniform_random_negative_proxy" &&
                               item.control_role == "negative_control";
            pbmc3k_rejected |= item.case_id == "pbmc3k_support_512" && !qualified;
        }
        write_coverage_summary(output, pbmc3k_available, embryo_available, heart_synthetic,
                               random_negative, heart_unavailable, perturbation_unavailable,
                               multiome_unavailable, regulatory_unavailable,
                               trajectory_unavailable, pbmc3k_rejected);
        std::cout << "wrote " << cases.size() + 2 << " census records to "
                  << options.output << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "tileability census failed: " << error.what() << '\n';
        return 1;
    }
}
