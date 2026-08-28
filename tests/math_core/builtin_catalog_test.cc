#include <Cellerator/compute/operation/builtin_catalog.hh>

#include <cstdlib>
#include <cstring>
#include <iostream>

namespace {

namespace core = cellerator::compute::math::core;

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "builtin_catalog_test: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void require_same_registry(
    const core::candidate_registry &lhs,
    const core::candidate_registry &rhs,
    const char *message) {
    require(std::memcmp(&lhs, &rhs, sizeof(lhs)) == 0, message);
}

} // namespace

int main() {
    const core::operation_status valid =
        core::validate_built_in_candidate_catalog();
    require(static_cast<bool>(valid), "catalog validation failed");

    const core::built_in_candidate_catalog_view catalog =
        core::built_in_candidate_catalog();
    require(catalog.entries != nullptr, "catalog entries are null");
    require(catalog.size == core::builtin_candidate_count,
        "catalog size mismatch");

    for (std::uint32_t index = 0u; index < catalog.size; ++index) {
        const core::built_in_candidate_descriptor &entry =
            catalog.entries[index];
        const core::operation_candidate candidate = entry.factory();
        require(core::same_stable_id(entry.identity, candidate.identity),
            "factory identity mismatch");
        require(std::strcmp(entry.name, candidate.name) == 0,
            "factory name mismatch");
        require(core::find_built_in_candidate(entry.identity) == &entry,
            "catalog lookup mismatch");
        require(entry.minimum_dense_width <= entry.maximum_dense_width,
            "invalid dense-width regime");
        for (std::uint32_t previous = 0u; previous < index; ++previous)
            require(!core::same_stable_id(
                    entry.identity, catalog.entries[previous].identity),
                "catalog identity is not unique");
    }

    require(catalog.entries[0].preparation
            == core::preparation_family::row_masked_n1
        && catalog.entries[1].preparation
            == core::preparation_family::csr_n1
        && catalog.entries[2].preparation
            == core::preparation_family::feature_major_small_n
        && catalog.entries[3].preparation
            == core::preparation_family::feature_major_cta_medium_n
        && catalog.entries[4].preparation
            == core::preparation_family::transpose_backward_n1,
        "catalog order is not deterministic");
    require(catalog.entries[4].output_axis == core::catalog_output_axis::source,
        "transpose output axis is not explicit");
    require((catalog.entries[4].preparation_requirements
            & core::catalog_transpose_value_map) != 0u,
        "transpose value map requirement is missing");

    core::candidate_registry registry{};
    require(static_cast<bool>(
            core::register_built_in_candidate_catalog(&registry)),
        "aggregate registration failed");
    require(registry.size == catalog.size, "registered size mismatch");
    for (std::uint32_t index = 0u; index < registry.size; ++index)
        require(core::same_stable_id(
                registry.candidates[index].identity,
                catalog.entries[index].identity),
            "registration order mismatch");

    const core::candidate_registry duplicate_baseline = registry;
    const core::operation_status duplicate =
        core::register_built_in_candidate_catalog(&registry);
    require(duplicate.code == core::operation_status_code::duplicate_candidate,
        "duplicate registration did not fail closed");
    require_same_registry(registry, duplicate_baseline,
        "duplicate registration partially mutated registry");

    core::candidate_registry full{};
    const core::operation_candidate seed = catalog.entries[0].factory();
    full.size = core::operation_candidate_capacity
        - core::builtin_candidate_count + 1u;
    for (std::uint32_t index = 0u; index < full.size; ++index) {
        full.candidates[index] = seed;
        full.candidates[index].identity = {index + 1u, 0xfeedu};
    }
    const core::candidate_registry full_baseline = full;
    const core::operation_status capacity =
        core::register_built_in_candidate_catalog(&full);
    require(capacity.code == core::operation_status_code::registry_full,
        "capacity overflow was not rejected");
    require_same_registry(full, full_baseline,
        "capacity failure partially mutated registry");

    std::cout << "builtin_catalog_test passed candidates=" << catalog.size
              << '\n';
    return EXIT_SUCCESS;
}
