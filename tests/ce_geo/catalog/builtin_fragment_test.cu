#include <Cellerator/compute/operation/builtin_catalog.hh>

#include <cstdlib>
#include <cstring>
#include <iostream>

namespace core = cellerator::compute::math::core;

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "builtin_fragment_test: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

} // namespace

int main() {
    const core::built_in_candidate_catalog_view legacy =
        core::built_in_candidate_catalog();
    const core::candidate_catalog_fragment_v2 &fragment =
        core::built_in_candidate_catalog_fragment_v2();

    require(core::validate_candidate_catalog_fragment_v2(fragment)
            == core::candidate_catalog_status_v2::success,
        "compatibility fragment failed validation");
    require(fragment.entry_count == legacy.size,
        "compatibility fragment changed candidate count");
    require((fragment.flags & core::candidate_fragment_compatibility) != 0u,
        "compatibility status is not explicit");

    for (std::uint32_t index = 0u; index < legacy.size; ++index) {
        const core::built_in_candidate_descriptor &old_entry =
            legacy.entries[index];
        const core::candidate_descriptor_v2 &new_entry =
            fragment.entries[index];
        const core::operation_candidate old_candidate = old_entry.factory();
        require(core::same_stable_id(
                new_entry.candidate.identity, old_candidate.identity),
            "candidate identity changed");
        require(std::strcmp(new_entry.candidate.name, old_candidate.name) == 0,
            "candidate name changed");
        require(new_entry.candidate.operation == old_candidate.operation
                && new_entry.candidate.projection == old_candidate.projection
                && new_entry.candidate.backend == old_candidate.backend,
            "candidate dispatch contract changed");
        require(new_entry.candidate.supports_numeric
                == old_candidate.supports_numeric
                && new_entry.candidate.prepare == old_candidate.prepare,
            "candidate function identity changed");
        require(new_entry.minimum_dense_width == old_entry.minimum_dense_width
                && new_entry.maximum_dense_width
                    == old_entry.maximum_dense_width,
            "candidate dense-width regime changed");
    }

    core::candidate_registry registry{};
    require(static_cast<bool>(
            core::register_built_in_candidate_catalog(&registry)),
        "legacy registration behavior failed");
    require(registry.size == fragment.entry_count,
        "legacy registration count changed");

    std::cout << "builtin_fragment_test passed candidates="
              << fragment.entry_count << '\n';
    return EXIT_SUCCESS;
}
