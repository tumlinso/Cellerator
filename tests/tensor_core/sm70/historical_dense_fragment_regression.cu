#include <Cellerator/compute/candidate/tensor_core/v100_dense_fragment_candidate.hh>
#include <Cellerator/compute/candidate/tensor_core/v100_dense_fragment_plan.hh>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <type_traits>
#include <vector>

namespace tc = cellerator::compute::math::tensor_core;

namespace {

[[noreturn]] void fail(const char *message) {
    std::cerr << "historical_dense_fragment_regression: " << message << '\n';
    std::abort();
}

void require(bool condition, const char *message) {
    if (!condition) fail(message);
}

std::filesystem::path source_root() {
    if (const char *configured = std::getenv("CELLERATOR_SOURCE_DIR"))
        return configured;
    std::filesystem::path candidate = std::filesystem::current_path();
    for (unsigned depth = 0u; depth != 8u; ++depth) {
        if (std::filesystem::exists(candidate / "docs/CE_GEO_PROGRAM.md")
            && std::filesystem::exists(candidate / "bench/ce_live/tensor_core"))
            return candidate;
        if (!candidate.has_parent_path()) break;
        candidate = candidate.parent_path();
    }
    fail("could not locate Cellerator source root");
}

std::string read_text(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), "historical evidence file is missing");
    return {std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
}

void require_contains(const std::string &text, const char *needle,
    const char *message) {
    require(text.find(needle) != std::string::npos, message);
}

void preserve_exact_fragment_and_residual_ownership() {
    std::vector<std::uint64_t> offsets(18u, 0u);
    std::vector<std::uint32_t> indices;
    for (std::uint32_t destination = 0u; destination != 16u; ++destination) {
        for (std::uint32_t source = 0u; source != 8u; ++source)
            indices.push_back(source);
        offsets[destination + 1u] = indices.size();
    }
    indices.push_back(16u);
    offsets[17u] = indices.size();

    tc::destination_row_csr_support_view support{offsets.data(),
        indices.data(), 17u, 17u, indices.size()};
    std::vector<std::uint16_t> tile_nnz(4u);
    std::vector<std::int64_t> tile_to_fragment(4u);
    std::vector<std::uint32_t> destination_bases(1u);
    std::vector<std::uint32_t> source_bases(1u);
    std::vector<std::uint64_t> edge_to_slot(indices.size());
    std::vector<std::uint64_t> slot_to_edge(256u);
    tc::v100_dense_fragment_plan_buffers buffers{tile_nnz.data(),
        tile_to_fragment.data(), tile_nnz.size(), destination_bases.data(),
        source_bases.data(), destination_bases.size(), edge_to_slot.data(),
        edge_to_slot.size(), slot_to_edge.data(), slot_to_edge.size()};
    tc::v100_dense_fragment_plan_requirements requirements{};

    require(tc::build_v100_dense_fragment_plan_host(
            support, buffers, &requirements)
            == tc::dense_fragment_plan_status::ok,
        "historical dense-fragment plan no longer builds");
    require(requirements.qualified_fragment_count == 1u,
        "frozen 50-percent qualification changed");
    require(requirements.maximum_tile_nnz == 128u,
        "frozen qualification census changed");
    require(requirements.residual_edge_count == 1u,
        "tail edge no longer has explicit residual ownership");
    require(edge_to_slot.back() == tc::invalid_dense_fragment_position,
        "historical plan silently absorbed a residual edge");
    for (std::uint64_t edge = 0u; edge != 128u; ++edge) {
        const std::uint64_t slot = edge_to_slot[edge];
        require(slot != tc::invalid_dense_fragment_position
                && slot_to_edge[slot] == edge,
            "historical logical-edge map changed");
    }
}

void preserve_pbmc3k_negative_control(const std::filesystem::path &root) {
    const std::string decision = read_text(root /
        "bench/ce_live/tensor_core/campaign/v100_decision_v1.json");
    require_contains(decision, "\"decision\": \"measured_rejection\"",
        "PBMC3K dense-fragment rejection was rewritten");
    require_contains(decision, "\"qualified_fragment_tiles\": 0",
        "PBMC3K zero-qualified-fragment evidence changed");
    require_contains(decision, "\"maximum_tile_nnz\": 106",
        "PBMC3K maximum tile census changed");
    require_contains(decision, "\"registered_in_builtin_catalog\": false",
        "historical candidate was promoted in its decision artifact");
    require_contains(decision,
        "5ec566e0bd56b468e9025ffe7c75fc54a4cf0eae2bc93107ae570fae188a7ccb",
        "checksum-pinned PBMC3K structure identity changed");

    const std::string contract = read_text(root /
        "bench/ce_live/tensor_core/contract/v100_dense_fragment_candidate_v1.json");
    require_contains(contract, "\"cuda_arch\": \"sm_70\"",
        "historical architecture contract changed");
    require_contains(contract, "\"global_default\": false",
        "historical candidate became a default");
    require_contains(contract,
        "complete amortized cost does not beat the best legal baseline",
        "historical complete-cost rejection rule changed");

    const std::string catalog = read_text(root /
        "src/compute/operation/builtin_catalog.cc");
    require(catalog.find("v100_dense_fragment_candidate") == std::string::npos,
        "historical candidate entered the production built-in catalog");
    require(catalog.find("v100-wmma-dense-fragment-f16-f32")
            == std::string::npos,
        "historical candidate name entered the production built-in catalog");
}

} // namespace

int main() {
    static_assert(tc::v100_dense_fragment_schema_version == 1u);
    static_assert(tc::v100_dense_fragment_variant == 1u);
    static_assert(tc::v100_dense_fragment_extent == 16u);
    static_assert(tc::v100_dense_fragment_candidate_id.low
        == 0x763130305f776d6dull);
    static_assert(tc::v100_dense_fragment_candidate_id.high
        == 0x615f64665f763100ull);
    static_assert(std::is_trivially_copyable<
        tc::v100_dense_fragment_projection_view>::value);

    tc::v100_dense_fragment_projection_view projection{};
    require(projection.architecture_class == 70u,
        "historical candidate no longer identifies sm_70");
    preserve_exact_fragment_and_residual_ownership();
    preserve_pbmc3k_negative_control(source_root());
    return 0;
}
