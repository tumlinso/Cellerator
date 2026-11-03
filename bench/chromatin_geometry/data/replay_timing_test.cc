#include "data/replay.hh"

#include <openssl/evp.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unistd.h>
#include <vector>

namespace {

struct Expected final {
    std::string_view id;
    std::string_view output_sha256;
    std::string_view workload_sha256;
};

constexpr std::array<Expected, 10> expected{{
    {"dag7-legacy-producer-only-sparse-low-frozen-r1-change0",
     "9ecd3fcc14e01400bd56cce019d61cdba1040d14ff40cb0474730e9ca6aa2cb5",
     "d3a47df24ecd5990b0c2d5b00009d978323dfc14ca38b7e41c7f113051479840"},
    {"dag7-coherent-producer-only-dense-high-update-r16-change1",
     "efd63a85530626bc44e0633ca24fc09e8488acc38b741ed964ee93c3bbc83016",
     "9f78f124675d29d4e34fa0178e6b5094a225170a4b622a54cdc580688339739b"},
    {"dag7-diverse-producer-only-sparse-high-update-r256-change10",
     "855d4409e679a25afd8c0579cde2eb1aa3880492f2035b970768095625e51203",
     "4677987677c3cf83e8f0065e5c9f6e945addb73fa915901bce2183b7a4cfeaa8"},
    {"dag7-legacy-complete-consumer-sparse-low-frozen-r1-change0",
     "9278f269b1cf0a9d29182b27c795dea8802fbb89d5b0114081881ef002d32b9c",
     "7f5620612179f90da1aef68c36cdf02fe08f00690c0cbf806216593cad0ffc71"},
    {"dag7-64root-complete-consumer-coherent-dense-high-update-r16-change1",
     "57a4487a9b5205482e943baffd789fa46ea4cfe76db959a62f17b73fa3099289",
     "e51f3d099937847044565c10084416e8c26ed9cbdbea0b95d5e611cdefc734ba"},
    {"dag2304-legacy-producer-only-sparse-high-frozen-r256-change10",
     "c611512309379f95c3a5fd08960fd37fb0e686a75b9d7df92f1ef69041ca9996",
     "fde3e1834e8ac8fce8ea7bcd4c52745e320ef00df4d6090656b1e3e21efea342"},
    {"dag2304-diverse-producer-only-dense-low-update-r1-change100",
     "694c8d475daa426dce54e5ed3f68bbbcc4aece270bd1e43f762fdb93d1543a46",
     "d918b2340c0ccb614f37c8ff80ee5d498d2e9c63df3931cc3023106d102916ba"},
    {"dag2304-coherent-producer-only-dense-low-frozen-r16-change1",
     "27b46418b58fca1ec835e8b585187d9affcef115af6190f6de61036a58840adf",
     "978e7e4eacce6fc41da7098c7bcca9885ca2c4c0a61cff045c195908e1733479"},
    {"dag2304-legacy-complete-consumer-sparse-high-frozen-r256-change10",
     "9cab8cc17fd206c2995f114e8def24b8862d845cb96ac9462644c03ab42353c7",
     "a3124d40d852a572465b311e14a42ea585ce7af17162e22e06222487e3de278f"},
    {"dag2304-64root-complete-consumer-diverse-dense-low-update-r1-change100",
     "7972d27d900f9717467e2e7e0f53de3b5ec07f0719eacb007c0093cf2b672562",
     "3ed8dd59630f24c815f19a6273e855f11a34416725b3ff5a974bb52e981c98ee"},
}};

class Sha256 final {
public:
    Sha256() : context_(EVP_MD_CTX_new()) {
        if (!context_ || EVP_DigestInit_ex(context_, EVP_sha256(), nullptr) != 1)
            throw std::runtime_error("could not initialize SHA-256");
    }
    ~Sha256() { EVP_MD_CTX_free(context_); }
    Sha256(const Sha256&) = delete;
    Sha256& operator=(const Sha256&) = delete;

    void bytes(const void* data, std::size_t size) {
        if (size && EVP_DigestUpdate(context_, data, size) != 1)
            throw std::runtime_error("could not update SHA-256");
    }
    template<class T> void scalar(T value) {
        static_assert(std::is_trivially_copyable_v<T>);
        bytes(&value, sizeof(value));
    }
    template<class T> void vector(const std::vector<T>& values) {
        static_assert(std::is_trivially_copyable_v<T>);
        scalar(std::uint64_t(values.size()));
        bytes(values.data(), values.size() * sizeof(T));
    }
    std::string finish() {
        std::array<unsigned char, 32> digest{};
        unsigned size = 0;
        if (EVP_DigestFinal_ex(context_, digest.data(), &size) != 1 || size != digest.size())
            throw std::runtime_error("could not finish SHA-256");
        constexpr char digits[] = "0123456789abcdef";
        std::string result;
        result.reserve(64);
        for (const auto byte : digest) {
            result.push_back(digits[byte >> 4]);
            result.push_back(digits[byte & 15]);
        }
        return result;
    }

private:
    EVP_MD_CTX* context_;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
    if (!condition) fail(message);
}

std::string replay_output_sha256(const cgs::WorkloadFixture& fixture) {
    Sha256 hash;
    constexpr char domain[] = "cgs.replay.outputs.v1";
    hash.bytes(domain, sizeof(domain) - 1);
    hash.bytes(fixture.spec.id.data(), fixture.spec.id.size());
    hash.vector(fixture.accessibility);
    hash.vector(fixture.accessibility_after);
    hash.vector(fixture.predicates_before);
    hash.vector(fixture.predicates_after);
    hash.vector(fixture.changed_input_offsets);
    hash.vector(fixture.changed_input_features);
    hash.vector(fixture.edge_roots);
    hash.scalar(fixture.root_decision_changes);
    return hash.finish();
}

std::string workload_semantic_sha256(const cgs::WorkloadFixture& fixture) {
    Sha256 hash;
    constexpr char domain[] = "cgs.replay.workload-semantic.v1";
    // The baseline hash included the domain's terminating NUL.
    hash.bytes(domain, sizeof(domain));
    hash.bytes(fixture.spec.id.data(), fixture.spec.id.size());
    hash.vector(fixture.accessibility);
    hash.vector(fixture.accessibility_after);
    hash.vector(fixture.edge_roots);
    hash.vector(fixture.group_program);
    hash.vector(fixture.group_feature_offsets);
    hash.vector(fixture.group_features);
    hash.vector(fixture.group_output_offsets);
    hash.vector(fixture.group_outputs);
    for (const auto& bank : fixture.programs) {
        hash.scalar(std::uint64_t(bank.nodes.size()));
        for (const auto& node : bank.nodes) {
            hash.scalar(std::uint8_t(node.op));
            hash.scalar(node.feature);
            hash.scalar(node.first_child);
            hash.scalar(node.child_count);
        }
    }
    return hash.finish();
}

bool same_dag(const cgs::ConditionDAG& lhs, const cgs::ConditionDAG& rhs) {
    if (lhs.levels != rhs.levels || lhs.roots != rhs.roots || lhs.root != rhs.root ||
        lhs.nodes.size() != rhs.nodes.size())
        return false;
    for (std::size_t i = 0; i < lhs.nodes.size(); ++i) {
        const auto& a = lhs.nodes[i];
        const auto& b = rhs.nodes[i];
        if (a.op != b.op || a.feature != b.feature || a.first_child != b.first_child ||
            a.child_count != b.child_count)
            return false;
    }
    return true;
}

bool same_fixture(const cgs::WorkloadFixture& lhs, const cgs::WorkloadFixture& rhs) {
    if (lhs.spec.id != rhs.spec.id || lhs.spec.seed != rhs.spec.seed ||
        lhs.spec.dag_nodes != rhs.spec.dag_nodes || lhs.spec.cells != rhs.spec.cells ||
        lhs.spec.features != rhs.spec.features || lhs.spec.groups != rhs.spec.groups ||
        lhs.spec.condition_edges != rhs.spec.condition_edges ||
        lhs.edge_roots != rhs.edge_roots || lhs.demanded_roots != rhs.demanded_roots ||
        lhs.demanded_nodes != rhs.demanded_nodes || lhs.demanded_levels != rhs.demanded_levels ||
        lhs.predicate_features != rhs.predicate_features ||
        lhs.accessibility != rhs.accessibility || lhs.accessibility_after != rhs.accessibility_after ||
        lhs.predicates_before != rhs.predicates_before || lhs.predicates_after != rhs.predicates_after ||
        lhs.changed_input_offsets != rhs.changed_input_offsets ||
        lhs.changed_input_features != rhs.changed_input_features ||
        lhs.root_decision_changes != rhs.root_decision_changes ||
        lhs.dirty_node_offsets != rhs.dirty_node_offsets || lhs.dirty_nodes != rhs.dirty_nodes ||
        lhs.group_program != rhs.group_program ||
        lhs.group_feature_offsets != rhs.group_feature_offsets ||
        lhs.group_features != rhs.group_features ||
        lhs.group_output_offsets != rhs.group_output_offsets ||
        lhs.group_outputs != rhs.group_outputs || lhs.distal_memberships != rhs.distal_memberships ||
        lhs.programs.size() != rhs.programs.size())
        return false;
    for (std::size_t i = 0; i < lhs.programs.size(); ++i)
        if (!same_dag(lhs.programs[i], rhs.programs[i])) return false;
    return true;
}

class TemporaryDirectory final {
public:
    TemporaryDirectory() {
        for (unsigned attempt = 0; attempt != 100; ++attempt) {
            path_ = std::filesystem::temp_directory_path() /
                ("cgs-replay-timing-" + std::to_string(::getpid()) + "-" +
                 std::to_string(attempt));
            std::error_code error;
            if (std::filesystem::create_directory(path_, error)) return;
        }
        fail("could not create replay test temporary directory");
    }
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

void write_corrupt_payload(const std::filesystem::path& source,
                           const std::filesystem::path& destination) {
    std::filesystem::copy_file(source, destination);
    std::fstream file(destination, std::ios::in | std::ios::out | std::ios::binary);
    require(bool(file), "could not open replay mismatch fixture");
    file.seekg(100);
    char byte = 0;
    file.read(&byte, 1);
    require(bool(file), "could not read replay mismatch fixture");
    byte ^= 1;
    file.seekp(100);
    file.write(&byte, 1);
    require(bool(file), "could not corrupt replay mismatch fixture");
}

void write_truncated_payload(const std::filesystem::path& source,
                             const std::filesystem::path& destination) {
    std::ifstream input(source, std::ios::binary);
    std::ofstream output(destination, std::ios::binary);
    require(bool(input) && bool(output), "could not open replay truncation fixture");
    std::array<char, 80> prefix{};
    input.read(prefix.data(), prefix.size());
    require(input.gcount() == static_cast<std::streamsize>(prefix.size()),
            "could not read replay truncation prefix");
    output.write(prefix.data(), prefix.size());
    require(bool(output), "could not write replay truncation fixture");
}

void require_rejection_without_mutation(const std::filesystem::path& path,
                                        const std::string& expected_message) {
    auto fixture = cgs::materialize_workload(cgs::workload_manifest().front());
    const auto before = fixture;
    cgs::ReplayPreparationTimes timings{11.0, 12.0, 13, 14};
    bool rejected = false;
    try {
        cgs::apply_atac_replay(fixture, path.string(), &timings);
    } catch (const std::runtime_error& error) {
        rejected = true;
        require(std::string_view(error.what()).find(expected_message) != std::string_view::npos,
                "replay rejection message changed: " + std::string(error.what()));
    }
    require(rejected, "replay accepted an invalid payload");
    require(same_fixture(fixture, before), "failed replay mutated its workload fixture");
    require(timings.operational_input_load_ms == 11.0 &&
            timings.reference_recompute_ms == 12.0 &&
            timings.payload_bytes_hashed == 13 && timings.payload_bytes_read == 14,
            "failed replay changed the caller's timing result");
}

void test_manifest(const std::string& payload_path) {
    const auto& manifest = cgs::workload_manifest();
    require(manifest.size() == expected.size(), "frozen manifest count changed");
    constexpr std::uint64_t payload_size = 95881576;
    constexpr std::uint64_t cells = 12016;
    constexpr std::uint64_t words_per_peak = (cells + 31) / 32;

    for (std::size_t i = 0; i < manifest.size(); ++i) {
        require(manifest[i].id == expected[i].id, "frozen manifest order or identity changed");
        auto legacy = cgs::materialize_workload(manifest[i]);
        auto timed = cgs::materialize_workload(manifest[i]);
        cgs::apply_atac_replay(legacy, payload_path);
        cgs::ReplayPreparationTimes timings{};
        cgs::apply_atac_replay(timed, payload_path, &timings);

        require(same_fixture(legacy, timed), "timed overload changed replay outputs or fixture state");
        require(replay_output_sha256(timed) == expected[i].output_sha256,
                "replay output differs from the pre-change loader: " + manifest[i].id);
        require(workload_semantic_sha256(timed) == expected[i].workload_sha256,
                "workload semantic hash differs from the pre-change loader: " + manifest[i].id);
        require(std::isfinite(timings.operational_input_load_ms) &&
                    timings.operational_input_load_ms > 0.0,
                "operational replay timing is not positive and finite");
        require(std::isfinite(timings.reference_recompute_ms) &&
                    timings.reference_recompute_ms > 0.0,
                "reference recomputation timing is not positive and finite");
        require(timings.payload_bytes_hashed == payload_size,
                "hashed-byte count does not equal pinned payload size");
        require(timings.payload_bytes_read == 72 +
                    std::uint64_t(manifest[i].features) * words_per_peak * 4,
                "selected-read byte count does not match header and packed peak planes");
    }
}

void test_invalid_payloads(const std::string& payload_path) {
    TemporaryDirectory temporary;
    const auto mismatch = temporary.path() / "mismatch.cgatac";
    const auto truncated = temporary.path() / "truncated.cgatac";
    write_corrupt_payload(payload_path, mismatch);
    write_truncated_payload(payload_path, truncated);
    require_rejection_without_mutation(mismatch, "SHA-256 mismatch");
    require_rejection_without_mutation(truncated, "SHA-256 mismatch");
    require_rejection_without_mutation(temporary.path() / "missing.cgatac",
                                       "cannot open replay payload");
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) fail("usage: cgs_replay_timing_test <accessibility.cgatac>");
        test_manifest(argv[1]);
        test_invalid_payloads(argv[1]);
        std::cout << "replay timing contract verified across " << expected.size()
                  << " manifest fixtures\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
