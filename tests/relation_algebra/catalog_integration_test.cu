#include <Cellerator/compute/operation/relation_algebra_catalog.hh>

#include <cstdlib>
#include <iostream>

namespace operation = cellerator::compute::operation;
namespace core = cellerator::compute::math::core;

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "catalog_integration_test: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

} // namespace

int main() {
    require(static_cast<std::uint16_t>(core::operation_kind::sparse_dense_multiply)
            == 1u
            && static_cast<std::uint16_t>(
                   core::operation_kind::weighted_relation_reduce) == 2u
            && static_cast<std::uint16_t>(
                   core::operation_kind::sequence_predicate_accumulate) == 3u,
        "frozen operation-core v1 kinds changed");

    const operation::relation_algebra_catalog_view_v1 catalog =
        operation::relation_algebra_candidate_catalog_v1();
    require(catalog.entries != nullptr
            && catalog.entry_count
                == operation::relation_algebra_catalog_entry_count_v1,
        "relation entry view is incomplete");
    require(catalog.fragments != nullptr
            && catalog.fragment_count
                == operation::relation_algebra_catalog_fragment_count_v1,
        "candidate fragment view is incomplete");
    require(static_cast<bool>(
            operation::validate_relation_algebra_candidate_catalog_v1()),
        "relation catalog failed integrated validation");

    std::uint32_t descriptor_count = 0u;
    for (std::uint32_t fragment_index = 0u;
         fragment_index < catalog.fragment_count; ++fragment_index) {
        const core::candidate_catalog_fragment_v2 &fragment =
            catalog.fragments[fragment_index];
        require(core::validate_candidate_catalog_fragment_v2(fragment)
                == core::candidate_catalog_status_v2::success,
            "candidate-catalog-v2 fragment is not exact");
        descriptor_count += fragment.entry_count;
    }
    require(descriptor_count == catalog.entry_count,
        "relation and candidate catalog cardinalities differ");

    for (std::uint32_t index = 0u; index < catalog.entry_count; ++index) {
        const operation::relation_algebra_catalog_entry_v1 &entry =
            catalog.entries[index];
        const core::candidate_descriptor_v2 *candidate =
            operation::find_relation_algebra_candidate_v2(entry.relation_kind);
        require(candidate != nullptr
                && core::same_stable_id(
                    candidate->candidate.identity, entry.candidate_identity),
            "relation entry did not resolve its candidate");
        require(candidate->projection_contract.variant
                == static_cast<std::uint16_t>(entry.relation_kind),
            "projection contract lost the relation kind");

        core::prepared_operation sentinel{};
        sentinel.schema_version = 99u;
        sentinel.run = reinterpret_cast<core::run_function>(
            static_cast<std::uintptr_t>(1u));
        const core::operation_status status = candidate->candidate.prepare(
            candidate->candidate, {}, {}, {}, {}, {}, &sentinel);
        require(status.code == core::operation_status_code::unsupported_problem,
            "declarative relation candidate did not fail closed");
        require(sentinel.schema_version == core::operation_core_schema_version
                && sentinel.run == nullptr,
            "failed preparation retained stale dispatch state");

        if (index < 2u) {
            require(entry.compatibility
                    == operation::operation_core_compatibility_v1::direct_schema_v1
                    && entry.required_operation_core_schema
                        == core::operation_core_schema_version
                    && candidate->candidate.operation
                        == core::operation_kind::sparse_dense_multiply,
                "direct apply mapping reinterpreted operation-core v1");
        } else {
            const std::uint16_t encoding =
                static_cast<std::uint16_t>(candidate->candidate.operation);
            require(entry.compatibility
                    == operation::operation_core_compatibility_v1::requires_schema_v2
                    && entry.required_operation_core_schema
                        == operation::relation_algebra_operation_core_schema_v2,
                "new relation operation escaped its schema-v2 gate");
            require(encoding == 0x1000u
                        + static_cast<std::uint16_t>(entry.relation_kind)
                    && encoding > static_cast<std::uint16_t>(
                        core::operation_kind::sequence_predicate_accumulate),
                "new relation operation collided with a frozen v1 kind");
        }
    }

    core::candidate_catalog_fragment_v2 malformed = catalog.fragments[1];
    malformed.entries = nullptr;
    require(core::validate_candidate_catalog_fragment_v2(malformed)
            == core::candidate_catalog_status_v2::invalid_fragment,
        "malformed candidate-catalog-v2 fragment was accepted");

    require(operation::find_relation_algebra_catalog_entry_v1(
                static_cast<operation::relation_algebra_kind_v1>(99u)) == nullptr
            && operation::find_relation_algebra_candidate_v2(
                static_cast<operation::relation_algebra_kind_v1>(99u)) == nullptr,
        "unknown relation kind resolved through the catalog");

    std::cout << "catalog_integration_test passed entries="
              << catalog.entry_count << " fragments=" << catalog.fragment_count
              << '\n';
    return EXIT_SUCCESS;
}
