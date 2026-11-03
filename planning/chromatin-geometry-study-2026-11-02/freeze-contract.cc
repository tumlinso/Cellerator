// Host-only deterministic exporter for the workload and validator freeze.
// Build with workloads.cc; this program does not link or initialize CUDA.
#include "workloads.hh"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace fs = std::filesystem;
namespace {

using Digest = std::array<unsigned char, 32>;

Digest digest(const unsigned char* bytes, std::size_t count) {
    Digest out{};
    unsigned written = 0;
    if (EVP_Digest(bytes, count, out.data(), &written, EVP_sha256(), nullptr) != 1 ||
        written != out.size())
        throw std::runtime_error("OpenSSL SHA-256 failed");
    return out;
}

std::string hex(const Digest& value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(value.size() * 2);
    for (const auto byte : value) {
        out.push_back(digits[byte >> 4]);
        out.push_back(digits[byte & 15]);
    }
    return out;
}

Digest digest_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path.string());
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) throw std::runtime_error("OpenSSL context allocation failed");
    const bool initialized = EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) == 1;
    std::array<char, 1 << 16> buffer{};
    bool ok = initialized;
    while (ok && in) {
        in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto got = in.gcount();
        if (got > 0)
            ok = EVP_DigestUpdate(ctx, buffer.data(), static_cast<std::size_t>(got)) == 1;
    }
    Digest result{};
    unsigned size = 0;
    ok = ok && in.eof() && EVP_DigestFinal_ex(ctx, result.data(), &size) == 1 &&
         size == result.size();
    EVP_MD_CTX_free(ctx);
    if (!ok) throw std::runtime_error("OpenSSL SHA-256 read failed for " + path.string());
    return result;
}

Digest digest_concatenated_files(const std::vector<fs::path>& paths) {
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) throw std::runtime_error("OpenSSL context allocation failed");
    bool ok = EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) == 1;
    std::array<char, 1 << 16> buffer{};
    for (const auto& path : paths) {
        std::ifstream in(path, std::ios::binary);
        if (!in) { ok = false; break; }
        while (ok && in) {
            in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
            const auto got = in.gcount();
            if (got > 0)
                ok = EVP_DigestUpdate(ctx, buffer.data(), static_cast<std::size_t>(got)) == 1;
        }
        if (!in.eof()) ok = false;
        if (!ok) break;
    }
    Digest result{};
    unsigned size = 0;
    ok = ok && EVP_DigestFinal_ex(ctx, result.data(), &size) == 1 &&
         size == result.size();
    EVP_MD_CTX_free(ctx);
    if (!ok) throw std::runtime_error("OpenSSL concatenated-file SHA-256 failed");
    return result;
}

template <class T>
Digest digest_values(const std::vector<T>& values) {
    static_assert(std::is_integral_v<T>);
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) throw std::runtime_error("OpenSSL context allocation failed");
    bool ok = EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) == 1;
    std::array<unsigned char, sizeof(T)> encoded{};
    for (const T value : values) {
        using U = std::make_unsigned_t<T>;
        const U bits = static_cast<U>(value);
        for (std::size_t i = 0; i < sizeof(T); ++i)
            encoded[i] = static_cast<unsigned char>((bits >> (i * 8)) & U{0xff});
        ok = ok && EVP_DigestUpdate(ctx, encoded.data(), encoded.size()) == 1;
    }
    Digest result{};
    unsigned size = 0;
    ok = ok && EVP_DigestFinal_ex(ctx, result.data(), &size) == 1 && size == result.size();
    EVP_MD_CTX_free(ctx);
    if (!ok) throw std::runtime_error("OpenSSL array SHA-256 failed");
    return result;
}

std::string quote(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char ch : value) {
        switch (ch) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (ch < 0x20) {
                out << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                    << static_cast<unsigned>(ch) << std::dec;
            } else out << static_cast<char>(ch);
        }
    }
    out << '"';
    return out.str();
}

const char* name(cgs::ProgramPattern value) {
    switch (value) {
    case cgs::ProgramPattern::coherent: return "coherent";
    case cgs::ProgramPattern::diverse: return "diverse";
    }
    throw std::runtime_error("unrecognized program pattern");
}
const char* name(cgs::ActivityPattern value) {
    switch (value) {
    case cgs::ActivityPattern::sparse: return "sparse";
    case cgs::ActivityPattern::dense: return "dense";
    }
    throw std::runtime_error("unrecognized activity pattern");
}
const char* name(cgs::OverlapPattern value) {
    switch (value) {
    case cgs::OverlapPattern::low: return "low";
    case cgs::OverlapPattern::high: return "high";
    }
    throw std::runtime_error("unrecognized overlap pattern");
}
const char* name(cgs::ParameterPattern value) {
    switch (value) {
    case cgs::ParameterPattern::frozen: return "frozen";
    case cgs::ParameterPattern::update_every_step: return "update_every_step";
    }
    throw std::runtime_error("unrecognized parameter pattern");
}
const char* name(cgs::WorkloadRole value) {
    switch (value) {
    case cgs::WorkloadRole::complete_consumer: return "complete_consumer";
    case cgs::WorkloadRole::producer_only_diagnostic: return "producer_only_diagnostic";
    }
    throw std::runtime_error("unrecognized workload role");
}

struct FileFact { std::string path; std::string sha; std::uintmax_t size; };

const std::vector<std::string> kContractFiles{
    "bench/chromatin_geometry/docs/draft.md",
    "planning/chromatin-geometry-study-2026-11-02/protocol.md",
    "bench/chromatin_geometry/contract.hh",
    "bench/chromatin_geometry/workloads.hh",
    "bench/chromatin_geometry/workloads.cc",
    "bench/chromatin_geometry/reference/CONTRACT.md",
    "bench/chromatin_geometry/reference/reference.hh",
    "bench/chromatin_geometry/reference/reference.cc",
    "tests/framework/cuda/grouped_linear_smoke.cu",
    "bench/chromatin_geometry/reference/reference_checks.cc",
    "bench/chromatin_geometry/reference/reference_smoke.cc",
    "bench/chromatin_geometry/reference/teacher_reference.hh",
    "bench/chromatin_geometry/contract_smoke.cc",
    "bench/chromatin_geometry/device_fixture.hh",
    "bench/chromatin_geometry/teacher_fixture.hh",
    "bench/chromatin_geometry/backends/consumer_fixture.hh",
    "bench/chromatin_geometry/backends/consumer_fixture.cc",
    "bench/chromatin_geometry/backends/conditions.hh",
    "bench/chromatin_geometry/backends/descriptor_debug.hh",
    "bench/chromatin_geometry/backends/consumer_tests.cu",
    "bench/chromatin_geometry/condition_tests_main.cc",
    "bench/chromatin_geometry/consumer_tests_main.cc",
    "bench/chromatin_geometry/learning/ste.hh",
    "bench/chromatin_geometry/learning/ste_chain.hh",
    "bench/chromatin_geometry/learning/ste_tests.cu",
    "bench/chromatin_geometry/learning/ste_host_tests.cc",
    "bench/chromatin_geometry/learning/ste_gpu_test_main.cc",
    "bench/chromatin_geometry/learning/model_update.hh",
    "bench/chromatin_geometry/learning/teacher.hh",
    "bench/chromatin_geometry/learning/teacher_chain.hh",
    "bench/chromatin_geometry/learning/teacher_test.cu",
    "bench/chromatin_geometry/learning/teacher_chain_test.cu",
    "bench/chromatin_geometry/learning/teacher_host_test.cc",
    "bench/chromatin_geometry/tests/test_repro_controls.py",
    "bench/chromatin_geometry/tests/verify_replay_source.cc",
    "bench/chromatin_geometry/data/FORMAT.md",
    "bench/chromatin_geometry/data/prepare_atac.cu",
    "bench/chromatin_geometry/data/replay.hh",
    "bench/chromatin_geometry/data/replay.cc",
    "bench/chromatin_geometry/data/replay_timing_test.cc",
    "bench/chromatin_geometry/data/verify_atac.cc",
    "bench/chromatin_geometry/campaign_runner.hh",
    "bench/chromatin_geometry/campaign_runner.cu",
    "bench/chromatin_geometry/campaign_layout_test.cc",
    "bench/chromatin_geometry/data/cache/pbmc-atac-cgatac01/manifest.json",
    "components/CellShard/include/CellShard/ingest/internal/tenx_h5/tenx_h5_reader.cuh",
    "planning/chromatin-geometry-study-2026-11-02/freeze-contract.cc",
    "planning/chromatin-geometry-study-2026-11-02/archive/freeze-contract-v1.cc",
};

std::vector<FileFact> collect_source_facts() {
    std::vector<FileFact> facts;
    for (const auto& rel : kContractFiles) {
        const fs::path path(rel);
        if (!fs::is_regular_file(path) || fs::is_symlink(path))
            throw std::runtime_error("freeze input is missing or not a regular file: " + rel);
        facts.push_back({rel, hex(digest_file(path)), fs::file_size(path)});
    }
    return facts;
}

std::string field(const std::string& json, const std::string& key) {
    const std::string marker = "\"" + key + "\"";
    auto at = json.find(marker);
    if (at == std::string::npos) throw std::runtime_error("manifest missing " + key);
    at = json.find(':', at + marker.size());
    if (at == std::string::npos) throw std::runtime_error("manifest malformed " + key);
    at = json.find('"', at + 1);
    if (at == std::string::npos) throw std::runtime_error("manifest field not a string " + key);
    const auto end = json.find('"', at + 1);
    if (end == std::string::npos) throw std::runtime_error("manifest field unterminated " + key);
    return json.substr(at + 1, end - at - 1);
}

std::uint64_t integer_field(const std::string& json, const std::string& key) {
    const std::string marker = "\"" + key + "\"";
    auto at = json.find(marker);
    if (at == std::string::npos) throw std::runtime_error("manifest missing " + key);
    at = json.find(':', at + marker.size());
    if (at == std::string::npos) throw std::runtime_error("manifest malformed " + key);
    ++at;
    while (at < json.size() && std::isspace(static_cast<unsigned char>(json[at]))) ++at;
    std::uint64_t value = 0;
    std::size_t read = 0;
    try { value = std::stoull(json.substr(at), &read); }
    catch (...) { throw std::runtime_error("manifest field not an integer " + key); }
    if (read == 0) throw std::runtime_error("manifest field not an integer " + key);
    return value;
}

template <class T>
std::uint64_t count_ones(const std::vector<T>& values) {
    return static_cast<std::uint64_t>(std::count(values.begin(), values.end(), T{1}));
}

struct OverlapStats {
    std::uint64_t pairs = 0;
    std::uint64_t overlapping_pairs = 0;
    std::uint64_t shared_memberships = 0;
    std::uint64_t maximum_intersection = 0;
    double mean_jaccard = 0.0;
};

OverlapStats overlap_stats(const std::vector<std::uint64_t>& offsets,
                           const std::vector<std::uint64_t>& members,
                           std::uint64_t groups) {
    OverlapStats stats;
    double jaccard_sum = 0.0;
    for (std::uint64_t region = 0; region < groups; region += 8) {
        const auto last = std::min(groups, region + 8);
        for (std::uint64_t lhs = region; lhs < last; ++lhs) {
            std::set<std::uint64_t> left(
                members.begin() + static_cast<std::ptrdiff_t>(offsets[lhs]),
                members.begin() + static_cast<std::ptrdiff_t>(offsets[lhs + 1]));
            for (std::uint64_t rhs = lhs + 1; rhs < last; ++rhs) {
                std::set<std::uint64_t> right(
                    members.begin() + static_cast<std::ptrdiff_t>(offsets[rhs]),
                    members.begin() + static_cast<std::ptrdiff_t>(offsets[rhs + 1]));
                std::uint64_t intersection = 0;
                for (const auto item : left) intersection += right.count(item) != 0;
                const auto union_count = left.size() + right.size() - intersection;
                ++stats.pairs;
                if (intersection) ++stats.overlapping_pairs;
                stats.shared_memberships += intersection;
                stats.maximum_intersection = std::max(stats.maximum_intersection, intersection);
                if (union_count) jaccard_sum += static_cast<double>(intersection) / union_count;
            }
        }
    }
    if (stats.pairs) stats.mean_jaccard = jaccard_sum / static_cast<double>(stats.pairs);
    return stats;
}

struct CohortStats {
    std::uint64_t measured_tiles = 0;
    std::uint64_t total_unique = 0;
    std::uint64_t minimum_unique = UINT64_MAX;
    std::uint64_t maximum_unique = 0;
    std::uint64_t minimum_cohort = UINT64_MAX;
    std::uint64_t maximum_cohort = 0;
    double entropy_sum = 0.0;
    std::vector<std::uint64_t> unique_histogram;
    std::vector<std::uint64_t> cohort_size_histogram;
    std::array<std::uint64_t, 256> signature_histogram{};
};

CohortStats cohort_stats(const cgs::WorkloadFixture& f, std::uint32_t tile_cells,
                         bool after_update = false) {
    CohortStats out;
    out.unique_histogram.resize(tile_cells + 1u);
    out.cohort_size_histogram.resize(tile_cells + 1u);
    const auto& s = f.spec;
    const auto& decisions = after_update ? f.predicates_after : f.predicates_before;
    const std::uint64_t neighborhood_count = (s.groups + 7u) / 8u;
    for (std::uint64_t neighborhood = 0; neighborhood < neighborhood_count; ++neighborhood) {
        const auto begin = neighborhood * 8u;
        const auto end = std::min<std::uint64_t>(s.groups, begin + 8u);
        for (std::uint64_t base_cell = 0; base_cell < s.cells; base_cell += tile_cells) {
            const auto tile_end = std::min<std::uint64_t>(s.cells, base_cell + tile_cells);
            std::array<std::uint32_t, 256> counts{};
            for (std::uint64_t cell = base_cell; cell < tile_end; ++cell) {
                std::uint32_t signature = 0;
                for (std::uint64_t group = begin; group < end; ++group) {
                    const auto edge = f.group_program[group];
                    if (edge >= s.condition_edges)
                        throw std::runtime_error("group edge exceeds frozen root map");
                    const auto flat = cell * s.condition_edges + edge;
                    if (decisions.at(flat))
                        signature |= 1u << static_cast<unsigned>(group - begin);
                }
                ++counts[signature];
            }
            std::uint64_t unique = 0, max_cohort = 0;
            double entropy = 0.0;
            const auto tile = tile_end - base_cell;
            for (std::size_t signature = 0; signature < counts.size(); ++signature) {
                const auto count = counts[signature];
                if (!count) continue;
                ++unique;
                out.minimum_cohort = std::min<std::uint64_t>(out.minimum_cohort, count);
                max_cohort = std::max<std::uint64_t>(max_cohort, count);
                const auto p = static_cast<double>(count) / tile;
                entropy -= p * std::log2(p);
                ++out.cohort_size_histogram.at(count);
                out.signature_histogram[signature] += count;
            }
            out.minimum_unique = std::min(out.minimum_unique, unique);
            out.maximum_unique = std::max(out.maximum_unique, unique);
            out.maximum_cohort = std::max(out.maximum_cohort, max_cohort);
            out.total_unique += unique;
            out.entropy_sum += entropy;
            ++out.measured_tiles;
            ++out.unique_histogram.at(unique);
        }
    }
    if (!out.measured_tiles) out.minimum_unique = 0;
    if (out.cohort_size_histogram.empty() ||
        std::all_of(out.cohort_size_histogram.begin(), out.cohort_size_histogram.end(),
                    [](std::uint64_t count) { return count == 0; }))
        out.minimum_cohort = 0;
    return out;
}

std::string stats_json(const OverlapStats& x) {
    std::ostringstream out;
    out << std::setprecision(9)
        << "{\"local_group_pairs\":" << x.pairs
        << ",\"overlapping_pairs\":" << x.overlapping_pairs
        << ",\"shared_memberships\":" << x.shared_memberships
        << ",\"maximum_pair_intersection\":" << x.maximum_intersection
        << ",\"mean_jaccard\":" << x.mean_jaccard << '}';
    return out.str();
}

std::string array_json(const std::array<std::uint64_t, 256>& values) {
    std::ostringstream out;
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; }
    out << ']';
    return out.str();
}

std::string array_json(const std::vector<std::uint64_t>& values) {
    std::ostringstream out;
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; }
    out << ']';
    return out.str();
}

std::string cohort_json(const CohortStats& x) {
    std::ostringstream out;
    out << std::setprecision(9)
        << "{\"measured_tiles\":" << x.measured_tiles
        << ",\"mean_unique_signatures\":"
        << (x.measured_tiles ? static_cast<double>(x.total_unique) / x.measured_tiles : 0.0)
        << ",\"minimum_unique_signatures\":" << x.minimum_unique
        << ",\"maximum_unique_signatures\":" << x.maximum_unique
        << ",\"minimum_nonempty_cohort_size\":" << x.minimum_cohort
        << ",\"maximum_cohort_size\":" << x.maximum_cohort
        << ",\"mean_entropy_bits\":"
        << (x.measured_tiles ? x.entropy_sum / x.measured_tiles : 0.0)
        << ",\"unique_signature_count_histogram\":" << array_json(x.unique_histogram)
        << ",\"cohort_size_histogram\":" << array_json(x.cohort_size_histogram)
        << ",\"signature_cell_count_histogram\":" << array_json(x.signature_histogram) << '}';
    return out.str();
}

std::string workload_json(const cgs::WorkloadSpec& spec) {
    const auto f = cgs::materialize_workload(spec);
    if (f.programs.size() != 1)
        throw std::runtime_error("shared-bank workload must contain one DAG");
    const auto& bank = f.programs.front();
    const auto before32 = cohort_stats(f, 32);
    const auto before64 = cohort_stats(f, 64);
    const auto before128 = cohort_stats(f, 128);
    const auto after32 = cohort_stats(f, 32, true);
    const auto after64 = cohort_stats(f, 64, true);
    const auto after128 = cohort_stats(f, 128, true);
    const auto participant_overlap = overlap_stats(f.group_feature_offsets, f.group_features,
                                                    spec.groups);
    const auto output_overlap = overlap_stats(f.group_output_offsets, f.group_outputs,
                                               spec.groups);
    const auto leaf_size = static_cast<std::uint64_t>(spec.cells) * f.predicate_features;
    const auto full_size = static_cast<std::uint64_t>(f.accessibility.size());
    const auto changes = static_cast<std::uint64_t>(f.changed_input_features.size());
    const auto root_size = static_cast<std::uint64_t>(f.predicates_before.size());
    std::vector<std::uint64_t> demanded_dirty_offsets{0u};
    std::vector<std::uint32_t> demanded_dirty_nodes;
    std::uint64_t dirty_cells = 0, demanded_dirty_cells = 0;
    for (std::uint64_t cell = 0; cell < spec.cells; ++cell) {
        const auto begin = f.dirty_node_offsets[cell];
        const auto end = f.dirty_node_offsets[cell + 1];
        dirty_cells += begin != end;
        const auto demanded_begin = demanded_dirty_nodes.size();
        for (auto at = begin; at < end; ++at) {
            const auto node = f.dirty_nodes[at];
            if (std::binary_search(f.demanded_nodes.begin(), f.demanded_nodes.end(), node))
                demanded_dirty_nodes.push_back(node);
        }
        demanded_dirty_cells += demanded_dirty_nodes.size() != demanded_begin;
        demanded_dirty_offsets.push_back(demanded_dirty_nodes.size());
    }
    const auto positive_before = count_ones(f.accessibility);
    const auto positive_after = count_ones(f.accessibility_after);
    std::uint64_t predicate_ones_before = 0, predicate_ones_after = 0;
    std::uint64_t leaf_positive = 0;
    for (std::uint64_t row = 0; row < spec.cells; ++row)
        for (std::uint64_t feature = 0; feature < f.predicate_features; ++feature)
            leaf_positive += f.accessibility[row * spec.features + feature] != 0;
    std::uint64_t distal_check = 0;
    std::vector<std::uint64_t> distal_members;
    for (std::uint64_t group = 0; group < spec.groups; ++group) {
        for (auto i = f.group_feature_offsets[group]; i < f.group_feature_offsets[group + 1]; ++i) {
            if (f.group_features[i] >= spec.features / 2u) {
                ++distal_check;
                distal_members.push_back(group);
                distal_members.push_back(i - f.group_feature_offsets[group]);
                distal_members.push_back(f.group_features[i]);
            }
        }
    }
    predicate_ones_before = count_ones(f.predicates_before);
    predicate_ones_after = count_ones(f.predicates_after);
    const auto leaf_size_bytes = std::vector<std::uint64_t>{spec.cells, spec.features,
                                                            f.predicate_features};
    const auto mapping_words = std::vector<std::uint64_t>(f.edge_roots.begin(), f.edge_roots.end());
    const auto group_programs = std::vector<std::uint64_t>(f.group_program.begin(), f.group_program.end());
    std::vector<std::uint64_t> bank_node_words;
    bank_node_words.reserve(bank.nodes.size() * 4u);
    for (const auto& node : bank.nodes) {
        bank_node_words.push_back(static_cast<std::uint64_t>(node.op));
        bank_node_words.push_back(node.feature);
        bank_node_words.push_back(node.first_child);
        bank_node_words.push_back(node.child_count);
    }
    const auto bank_root_words = std::vector<std::uint64_t>(bank.roots.begin(), bank.roots.end());
    const auto demanded_root_words = std::vector<std::uint64_t>(f.demanded_roots.begin(), f.demanded_roots.end());
    const auto demanded_node_words = std::vector<std::uint64_t>(f.demanded_nodes.begin(), f.demanded_nodes.end());
    std::vector<std::uint64_t> demanded_level_words;
    for (std::size_t level = 0; level < f.demanded_levels.size(); ++level) {
        demanded_level_words.push_back(level);
        demanded_level_words.push_back(f.demanded_levels[level].size());
        for (const auto node : f.demanded_levels[level]) demanded_level_words.push_back(node);
    }
    std::ostringstream out;
    out << std::setprecision(9)
        << "{\"id\":" << quote(spec.id)
        << ",\"seed\":" << spec.seed
        << ",\"role\":" << quote(name(spec.role))
        << ",\"program_pattern\":" << quote(name(spec.program_pattern))
        << ",\"activity_pattern\":" << quote(name(spec.activity_pattern))
        << ",\"overlap_pattern\":" << quote(name(spec.overlap_pattern))
        << ",\"parameters\":" << quote(name(spec.parameter_pattern))
        << ",\"update_policy\":" << quote(spec.parameter_pattern == cgs::ParameterPattern::frozen ? "frozen" : "update_every_step")
        << ",\"reuse_horizon\":" << spec.reuse_horizon
        << ",\"requested_leaf_change_percent\":" << unsigned(spec.predicate_change_percent)
        << ",\"cells\":" << spec.cells << ",\"features\":" << spec.features
        << ",\"predicate_features\":" << f.predicate_features
        << ",\"groups\":" << spec.groups << ",\"condition_edges\":" << spec.condition_edges
        << ",\"bank_nodes\":" << bank.nodes.size()
        << ",\"bank_roots\":" << bank.roots.size()
        << ",\"demanded_roots\":" << f.demanded_roots.size()
        << ",\"demanded_nodes\":" << f.demanded_nodes.size()
        << ",\"demanded_level_count\":" << f.demanded_levels.size()
        << ",\"legacy_root_map\":" << (spec.legacy_root_map ? "true" : "false")
        << ",\"leaf_changes\":" << changes
        << ",\"leaf_change_fraction\":"
        << (leaf_size ? static_cast<double>(changes) / leaf_size : 0.0)
        << ",\"actual_leaf_observations_changed\":" << changes
        << ",\"root_decision_changes\":" << f.root_decision_changes
        << ",\"root_decision_change_fraction\":"
        << (root_size ? static_cast<double>(f.root_decision_changes) / root_size : 0.0)
        << ",\"dirty_node_total\":" << f.dirty_nodes.size()
        << ",\"dirty_cells\":" << dirty_cells
        << ",\"mean_dirty_nodes_per_cell\":"
        << (spec.cells ? static_cast<double>(f.dirty_nodes.size()) / spec.cells : 0.0)
        << ",\"demanded_dirty_node_total\":" << demanded_dirty_nodes.size()
        << ",\"demanded_dirty_cells\":" << demanded_dirty_cells
        << ",\"mean_demanded_dirty_nodes_per_cell\":"
        << (spec.cells ? static_cast<double>(demanded_dirty_nodes.size()) / spec.cells : 0.0)
        << ",\"full_accessibility_positive_fraction_before\":"
        << (full_size ? static_cast<double>(positive_before) / full_size : 0.0)
        << ",\"full_accessibility_positive_fraction_after\":"
        << (full_size ? static_cast<double>(positive_after) / full_size : 0.0)
        << ",\"root_positive_fraction_before\":"
        << (root_size ? static_cast<double>(predicate_ones_before) / root_size : 0.0)
        << ",\"root_positive_fraction_after\":"
        << (root_size ? static_cast<double>(predicate_ones_after) / root_size : 0.0)
        << ",\"participant_overlap\":" << stats_json(participant_overlap)
        << ",\"output_overlap\":" << stats_json(output_overlap)
        << ",\"distal_memberships_declared\":" << f.distal_memberships
        << ",\"distal_memberships_observed\":" << distal_check
        << ",\"distal_mapping_sha256\":" << quote(hex(digest_values(distal_members)))
        << ",\"array_sha256\":{"
        << "\"accessibility_before\":" << quote(hex(digest_values(f.accessibility))) << ','
        << "\"accessibility_after\":" << quote(hex(digest_values(f.accessibility_after))) << ','
        << "\"root_decisions_before\":" << quote(hex(digest_values(f.predicates_before))) << ','
        << "\"root_decisions_after\":" << quote(hex(digest_values(f.predicates_after))) << ','
        << "\"edge_roots\":" << quote(hex(digest_values(mapping_words))) << ','
        << "\"group_programs\":" << quote(hex(digest_values(group_programs))) << ','
        << "\"bank_node_records_op_feature_first_child_child_count\":"
        << quote(hex(digest_values(bank_node_words))) << ','
        << "\"bank_roots\":" << quote(hex(digest_values(bank_root_words))) << ','
        << "\"demanded_roots\":" << quote(hex(digest_values(demanded_root_words))) << ','
        << "\"demanded_nodes\":" << quote(hex(digest_values(demanded_node_words))) << ','
        << "\"demanded_levels\":" << quote(hex(digest_values(demanded_level_words))) << ','
        << "\"changed_input_offsets\":" << quote(hex(digest_values(f.changed_input_offsets))) << ','
        << "\"changed_input_features\":" << quote(hex(digest_values(f.changed_input_features))) << ','
        << "\"dirty_node_offsets\":" << quote(hex(digest_values(f.dirty_node_offsets))) << ','
        << "\"dirty_nodes\":" << quote(hex(digest_values(f.dirty_nodes))) << ','
        << "\"demanded_dirty_node_offsets\":" << quote(hex(digest_values(demanded_dirty_offsets))) << ','
        << "\"demanded_dirty_nodes\":" << quote(hex(digest_values(demanded_dirty_nodes))) << ','
        << "\"group_feature_offsets\":" << quote(hex(digest_values(f.group_feature_offsets))) << ','
        << "\"group_features\":" << quote(hex(digest_values(f.group_features))) << ','
        << "\"group_output_offsets\":" << quote(hex(digest_values(f.group_output_offsets))) << ','
        << "\"group_outputs\":" << quote(hex(digest_values(f.group_outputs))) << ','
        << "\"shape\":" << quote(hex(digest_values(leaf_size_bytes))) << ','
        << "\"local_primal_signature_counts_before_tile32\":"
        << quote(hex(digest_values(std::vector<std::uint64_t>(before32.signature_histogram.begin(), before32.signature_histogram.end())))) << ','
        << "\"local_primal_signature_counts_before_tile64\":"
        << quote(hex(digest_values(std::vector<std::uint64_t>(before64.signature_histogram.begin(), before64.signature_histogram.end())))) << ','
        << "\"local_primal_signature_counts_before_tile128\":"
        << quote(hex(digest_values(std::vector<std::uint64_t>(before128.signature_histogram.begin(), before128.signature_histogram.end())))) << ','
        << "\"local_primal_signature_counts_after_tile32\":"
        << quote(hex(digest_values(std::vector<std::uint64_t>(after32.signature_histogram.begin(), after32.signature_histogram.end())))) << ','
        << "\"local_primal_signature_counts_after_tile64\":"
        << quote(hex(digest_values(std::vector<std::uint64_t>(after64.signature_histogram.begin(), after64.signature_histogram.end())))) << ','
        << "\"local_primal_signature_counts_after_tile128\":"
        << quote(hex(digest_values(std::vector<std::uint64_t>(after128.signature_histogram.begin(), after128.signature_histogram.end()))))
        << "},\"local_primal_cohorts_before_by_8_group_neighborhood\":{"
        << "\"tile_32\":" << cohort_json(before32) << ','
        << "\"tile_64\":" << cohort_json(before64) << ','
        << "\"tile_128\":" << cohort_json(before128) << "},"
        << "\"local_primal_cohorts_after_by_8_group_neighborhood\":{"
        << "\"tile_32\":" << cohort_json(after32) << ','
        << "\"tile_64\":" << cohort_json(after64) << ','
        << "\"tile_128\":" << cohort_json(after128) << "}"
        << ",\"predicate_activity_measured_over_declared_leaf_inputs\":"
        << (leaf_size ? static_cast<double>(leaf_positive) / leaf_size : 0.0)
        << ",\"predicate_activity_after_over_declared_leaf_inputs\":";
    std::uint64_t leaf_positive_after = 0;
    for (std::uint64_t row = 0; row < spec.cells; ++row)
        for (std::uint64_t feature = 0; feature < f.predicate_features; ++feature)
            leaf_positive_after += f.accessibility_after[row * spec.features + feature] != 0;
    out << (leaf_size ? static_cast<double>(leaf_positive_after) / leaf_size : 0.0) << '}';
    return out.str();
}

std::string replay_json() {
    const fs::path base("bench/chromatin_geometry/data/cache/pbmc-atac-cgatac01");
    const fs::path manifest_path = base / "manifest.json";
    std::ifstream in(manifest_path);
    if (!in) throw std::runtime_error("public replay manifest unavailable");
    const std::string manifest((std::istreambuf_iterator<char>(in)), {});
    const auto source_sha = field(manifest, "source_sha256");
    const auto replay_sha = field(manifest, "replay_sha256");
    const auto cells_sha = field(manifest, "cells_sha256");
    const auto features_sha = field(manifest, "features_sha256");
    const auto sidecars_sha = field(manifest, "identity_sidecars_sha256");
    const auto wrapper_sha = field(manifest, "wrapper_sha256");
    const auto cellshard_revision = field(manifest, "cellshard_revision");
    const auto source_path = fs::path("bench/chromatin_geometry/data/cache/") /
        "pbmc_unsorted_10k_filtered_feature_bc_matrix.h5";
    const auto replay_path = base / "accessibility.cgatac";
    const auto cells_path = base / "cells.tsv";
    const auto features_path = base / "features.tsv";
    for (const auto& path : {manifest_path, source_path, replay_path, cells_path, features_path})
        if (!fs::is_regular_file(path) || fs::is_symlink(path))
            throw std::runtime_error("public replay input is missing or not a regular file: " + path.string());
    const auto source_actual = hex(digest_file(source_path));
    const auto manifest_actual = hex(digest_file(manifest_path));
    const auto replay_actual = hex(digest_file(replay_path));
    const auto cells_actual = hex(digest_file(cells_path));
    const auto features_actual = hex(digest_file(features_path));
    const auto sidecars_actual = hex(digest_concatenated_files({cells_path, features_path}));
    const auto wrapper_actual = hex(digest_concatenated_files({replay_path, cells_path, features_path}));
    if (source_actual != source_sha || replay_actual != replay_sha || cells_actual != cells_sha ||
        features_actual != features_sha || sidecars_actual != sidecars_sha ||
        wrapper_actual != wrapper_sha)
        throw std::runtime_error("public replay manifest hash no longer matches its inputs");
    if (integer_field(manifest, "cells") != 12016 || integer_field(manifest, "peaks") != 63751)
        throw std::runtime_error("public replay dimensions differ from the reviewed dataset contract");
    constexpr std::uint64_t header_bytes = 72;
    const auto expected_payload_bytes = integer_field(manifest, "peaks") *
        integer_field(manifest, "words_per_peak") * sizeof(std::uint32_t);
    if (fs::file_size(replay_path) != header_bytes + expected_payload_bytes)
        throw std::runtime_error("prepared replay byte extent differs from its manifest dimensions");
    if (hex(digest_file(source_path)) != source_actual ||
        hex(digest_file(manifest_path)) != manifest_actual ||
        hex(digest_file(replay_path)) != replay_actual ||
        hex(digest_file(cells_path)) != cells_actual ||
        hex(digest_file(features_path)) != features_actual ||
        hex(digest_concatenated_files({cells_path, features_path})) != sidecars_actual ||
        hex(digest_concatenated_files({replay_path, cells_path, features_path})) != wrapper_actual)
        throw std::runtime_error("public replay source or preparation bytes changed during snapshot");
    return "{\"dataset\":" + quote(field(manifest, "dataset")) +
        ",\"release\":" + quote(field(manifest, "dataset_release")) +
        ",\"license\":" + quote(field(manifest, "license")) +
        ",\"source_url\":" + quote(field(manifest, "source_url")) +
        ",\"dataset_page\":" + quote(field(manifest, "dataset_page")) +
        ",\"source_sha256\":" + quote(source_actual) +
        ",\"source_bytes\":" + std::to_string(fs::file_size(source_path)) +
        ",\"manifest_sha256\":" + quote(manifest_actual) +
        ",\"replay_sha256\":" + quote(replay_actual) +
        ",\"prepared_payload_bytes_including_header\":" + std::to_string(fs::file_size(replay_path)) +
        ",\"packed_observation_payload_bytes\":" + std::to_string(expected_payload_bytes) +
        ",\"cells_sha256\":" + quote(cells_actual) +
        ",\"features_sha256\":" + quote(features_actual) +
        ",\"identity_sidecars_sha256\":" + quote(sidecars_actual) +
        ",\"wrapper_sha256\":" + quote(wrapper_actual) +
        ",\"payload_fnv64\":" + quote(std::to_string(integer_field(manifest, "payload_fnv64"))) +
        ",\"cellshard_revision\":" + quote(cellshard_revision) +
        ",\"native_reader_header_sha256\":" + quote(hex(digest_file("components/CellShard/include/CellShard/ingest/internal/tenx_h5/tenx_h5_reader.cuh"))) +
        ",\"preparation_compiler\":" + quote(field(manifest, "preparation_compiler")) +
        ",\"hdf5_version\":" + quote(field(manifest, "hdf5_version")) +
        ",\"cells\":" + std::to_string(integer_field(manifest, "cells")) +
        ",\"peaks\":" + std::to_string(integer_field(manifest, "peaks")) +
        ",\"observed_positive_only\":true,\"zero_means_biological_exclusion\":false}";
}

std::string main_json() {
    const auto before = collect_source_facts();
    const auto& manifest = cgs::workload_manifest();
    std::vector<std::string> cases;
    cases.reserve(manifest.size());
    for (const auto& spec : manifest) cases.push_back(workload_json(spec));
    const auto replay = replay_json();
    const auto after = collect_source_facts();
    if (before.size() != after.size()) throw std::runtime_error("source inventory changed during freeze");
    for (std::size_t i = 0; i < before.size(); ++i) {
        if (before[i].path != after[i].path || before[i].sha != after[i].sha ||
            before[i].size != after[i].size)
            throw std::runtime_error("freeze input changed during export: " + before[i].path);
    }
    const auto exporter_path = fs::path("planning/chromatin-geometry-study-2026-11-02/freeze-contract.cc");
    const auto exporter_sha = hex(digest_file(exporter_path));
    std::ostringstream out;
    out << std::setprecision(9)
        << "{\n\"format\":\"cgs.contract-freeze.v1\","
        << "\"cplusplus\":" << __cplusplus << ",\"compiler\":" << quote(__VERSION__)
        << ",\"exporter_build_command\":\"g++ -std=c++17 -O1 -Ibench/chromatin_geometry planning/chromatin-geometry-study-2026-11-02/freeze-contract.cc bench/chromatin_geometry/workloads.cc -lcrypto\""
        << ",\"exporter_sha256\":" << quote(exporter_sha)
        << ",\"array_hash_encoding\":\"SHA-256 over each integral value serialized little-endian in vector order; array shapes are recorded separately\""
        << ",\"local_primal_signature_definition\":\"For each contiguous 8-group neighborhood and cell tile, pack actual admission bits with bit i for group neighborhood_start+i; report per-tile unique signatures, base-2 mass entropy, and cohort-size histograms before and after leaf updates\""
        << ",\"source_files\":[";
    for (std::size_t i = 0; i < before.size(); ++i) {
        if (i) out << ',';
        out << "{\"path\":" << quote(before[i].path)
            << ",\"sha256\":" << quote(before[i].sha)
            << ",\"bytes\":" << before[i].size << '}';
    }
    out << "],\"product_bound\":{\"name\":\"product_abs_tolerance\","
        << "\"rule\":\"2e-4*(1+abs(reference))\",\"source\":\"bench/chromatin_geometry/reference/reference.hh::product_abs_tolerance and reference/CONTRACT.md\"},"
        << "\"grouped_linear_bound\":{\"kind\":\"per_element_absolute_compound\","
        << "\"dot_rule\":\"u=epsilon(float32)/2; k=2*dot_length*u; bound_dot=(2*k/(1-k))*sum(abs(products))+(FP16?fp16_ulp(expected):0)\","
        << "\"reduction_rule\":\"propagate local bounds and add fp32_reduction_bound(terms,sum_abs)=(2*k/(1-k))*sum_abs with k=2*terms*u\","
        << "\"source\":\"tests/framework/cuda/grouped_linear_smoke.cu::allowed_error; bench/chromatin_geometry/reference/reference.cc::grouped_linear_allowed_error and fp32_reduction_bound; reference/CONTRACT.md\","
        << "\"test_predicate\":\"within_absolute_bound; no relative fallback\"},"
        << "\"frozen_checks\":[\"native scalar reference and independent teacher row oracle\","
        << "\"bit-exact producer outputs, root mapping, tails and dirty-node coverage\","
        << "\"ordered products preserve repeated slots, zero participants, shared theta and overlapping outputs\","
        << "\"grouped linear validates fp16 staged F/dF/dX/dW and fp32 accumulation with frozen per-element bounds\","
        << "\"STE uses analytic full derivative membership including dormant candidates; no finite-difference estimator\","
        << "\"teacher hard table equals thresholded eight-row teacher; hard routes have zero task gate gradient\","
        << "\"excluded-forward poison reads, all-zero-arity, empty/tail and stale-revision guards\","
        << "\"Compute Sanitizer memory/race/init/sync test sources are fixed independently of performance reports\","
        << "\"public replay matches HDF5 source bits and stable cell/peak identities\"],"
        << "\"public_replay\":" << replay << ",\"workloads\":[";
    for (std::size_t i = 0; i < cases.size(); ++i) { if (i) out << ','; out << cases[i]; }
    out << "]}\n";
    return out.str();
}

} // namespace

int main(int argc, char** argv) {
    try {
        fs::path output("planning/chromatin-geometry-study-2026-11-02/contract-freeze-v1.json");
        if (argc == 3 && std::string(argv[1]) == "--output") output = argv[2];
        else if (argc != 1) throw std::runtime_error("usage: freeze-contract [--output PATH]");
        if (fs::exists(output))
            throw std::runtime_error("refusing to overwrite an existing contract freeze: " + output.string());
        const auto json = main_json();
        const auto tmp = output.string() + ".tmp";
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            if (!out) throw std::runtime_error("cannot create freeze temporary file");
            out.write(json.data(), static_cast<std::streamsize>(json.size()));
            out.flush();
            if (!out) throw std::runtime_error("cannot write freeze temporary file");
        }
        if (fs::exists(output)) throw std::runtime_error("contract freeze appeared during generation");
        // A same-directory hard link publishes the fully written inode
        // atomically and fails if another writer created the destination.
        // std::filesystem::rename may replace an existing file on POSIX.
        fs::create_hard_link(tmp, output);
        fs::remove(tmp);
        std::cout << hex(digest(reinterpret_cast<const unsigned char*>(json.data()), json.size()))
                  << "  " << output.generic_string() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "contract freeze: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
