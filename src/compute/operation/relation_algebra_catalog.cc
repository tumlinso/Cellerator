#include <Cellerator/compute/operation/relation_algebra_catalog.hh>

#include <cstring>

namespace cellerator::compute::operation {
namespace {

constexpr core::stable_id relation_algebra_provider_identity{
    0x72656c6174696f6eull, 0x2d616c6765627261ull};
constexpr core::stable_id direct_fragment_identity{
    0x72656c2d64697265ull, 0x63742d76312d7632ull};
constexpr core::stable_id schema_v2_fragment_identity{
    0x72656c2d73636865ull, 0x6d612d76322d7632ull};
constexpr core::stable_id projection_view_type{
    0x72656c2d616c6765ull, 0x6272612d76696577ull};

constexpr core::stable_id candidate_identity(
    relation_algebra_kind_v1 kind) noexcept {
    return {0x72656c2d63616e64ull,
        0x0000000000008600ull + static_cast<std::uint16_t>(kind)};
}

bool supports_valid_numeric(const core::numeric_policy &numeric) noexcept {
    return static_cast<bool>(core::validate_numeric_policy(numeric));
}

core::operation_status reject_declarative_candidate(
    const core::operation_candidate &,
    const core::operation_problem &,
    const core::structure_set_key &,
    const core::projection_key &,
    const core::numeric_policy &,
    const core::prepare_policy &,
    core::prepared_operation *prepared) noexcept {
    if (prepared != nullptr) *prepared = {};
    return {core::operation_status_code::unsupported_problem,
        execution::binding_validation_code::ok,
        "relation catalog declarations require typed candidate selection"};
}

core::operation_status reject_schema_v2_candidate(
    const core::operation_candidate &,
    const core::operation_problem &,
    const core::structure_set_key &,
    const core::projection_key &,
    const core::numeric_policy &,
    const core::prepare_policy &,
    core::prepared_operation *prepared) noexcept {
    if (prepared != nullptr) *prepared = {};
    return {core::operation_status_code::unsupported_problem,
        execution::binding_validation_code::ok,
        "relation operation requires operation-core schema v2"};
}

core::candidate_descriptor_v2 make_descriptor(
    relation_algebra_kind_v1 kind,
    const char *name,
    core::operation_kind operation,
    core::projection_kind projection,
    core::prepare_function prepare,
    std::uint32_t flags) noexcept {
    core::candidate_descriptor_v2 descriptor{};
    descriptor.candidate.identity = candidate_identity(kind);
    descriptor.candidate.name = name;
    descriptor.candidate.operation = operation;
    descriptor.candidate.projection = projection;
    descriptor.candidate.backend = core::backend_kind::composed;
    descriptor.candidate.capability_flags = core::candidate_deterministic
        | core::candidate_graph_capture;
    descriptor.candidate.supports_numeric = &supports_valid_numeric;
    descriptor.candidate.prepare = prepare;
    descriptor.provider_identity = relation_algebra_provider_identity;
    descriptor.projection_contract.view_type = projection_view_type;
    descriptor.projection_contract.abi_major = 1u;
    descriptor.projection_contract.schema_version =
        relation_algebra_schema_version_v1;
    descriptor.projection_contract.variant =
        static_cast<std::uint16_t>(kind);
    descriptor.flags = flags | core::candidate_descriptor_requires_measurement;
    return descriptor;
}

const core::candidate_descriptor_v2 direct_entries[2]{
    make_descriptor(relation_algebra_kind_v1::relation_apply,
        "relation-apply-v1-catalog-declaration",
        core::operation_kind::sparse_dense_multiply,
        core::projection_kind::csr,
        &reject_declarative_candidate,
        core::candidate_descriptor_compatibility),
    make_descriptor(relation_algebra_kind_v1::relation_apply_transpose,
        "relation-apply-transpose-v1-catalog-declaration",
        core::operation_kind::sparse_dense_multiply,
        core::projection_kind::transpose_or_backward,
        &reject_declarative_candidate,
        core::candidate_descriptor_compatibility)};

const core::candidate_descriptor_v2 schema_v2_entries[5]{
    make_descriptor(relation_algebra_kind_v1::contract_on_support,
        "contract-on-support-v2-catalog-declaration",
        operation_core_kind_v2(
            relation_algebra_operation_kind_v2::contract_on_support),
        core::projection_kind::architecture_specific,
        &reject_schema_v2_candidate, 0u),
    make_descriptor(relation_algebra_kind_v1::segment_reduce,
        "segment-reduce-v2-catalog-declaration",
        operation_core_kind_v2(relation_algebra_operation_kind_v2::segment_reduce),
        core::projection_kind::architecture_specific,
        &reject_schema_v2_candidate, 0u),
    make_descriptor(relation_algebra_kind_v1::segment_normalize,
        "segment-normalize-v2-catalog-declaration",
        operation_core_kind_v2(
            relation_algebra_operation_kind_v2::segment_normalize),
        core::projection_kind::architecture_specific,
        &reject_schema_v2_candidate, 0u),
    make_descriptor(relation_algebra_kind_v1::edge_map_or_gate,
        "edge-map-or-gate-v2-catalog-declaration",
        operation_core_kind_v2(
            relation_algebra_operation_kind_v2::edge_map_or_gate),
        core::projection_kind::architecture_specific,
        &reject_schema_v2_candidate, 0u),
    make_descriptor(relation_algebra_kind_v1::relation_bundle_apply,
        "relation-bundle-apply-v2-catalog-declaration",
        operation_core_kind_v2(
            relation_algebra_operation_kind_v2::relation_bundle_apply),
        core::projection_kind::architecture_specific,
        &reject_schema_v2_candidate, 0u)};

const core::candidate_catalog_fragment_v2 fragments[
    relation_algebra_catalog_fragment_count_v1]{
    {core::candidate_catalog_fragment_schema_version_v2,
        sizeof(core::candidate_catalog_fragment_v2),
        direct_fragment_identity,
        relation_algebra_provider_identity,
        "relation-algebra-direct-v1-catalog-v2",
        direct_entries,
        2u,
        core::candidate_fragment_builtin | core::candidate_fragment_compatibility,
        1u,
        {}},
    {core::candidate_catalog_fragment_schema_version_v2,
        sizeof(core::candidate_catalog_fragment_v2),
        schema_v2_fragment_identity,
        relation_algebra_provider_identity,
        "relation-algebra-schema-v2-catalog-v2",
        schema_v2_entries,
        5u,
        core::candidate_fragment_builtin,
        1u,
        {}}};

const relation_algebra_catalog_entry_v1 entries[
    relation_algebra_catalog_entry_count_v1]{
    {relation_algebra_catalog_schema_version_v1,
        sizeof(relation_algebra_catalog_entry_v1),
        relation_algebra_kind_v1::relation_apply,
        operation_core_compatibility_v1::direct_schema_v1,
        {}, core::operation_core_schema_version,
        candidate_identity(relation_algebra_kind_v1::relation_apply), {}},
    {relation_algebra_catalog_schema_version_v1,
        sizeof(relation_algebra_catalog_entry_v1),
        relation_algebra_kind_v1::relation_apply_transpose,
        operation_core_compatibility_v1::direct_schema_v1,
        {}, core::operation_core_schema_version,
        candidate_identity(relation_algebra_kind_v1::relation_apply_transpose), {}},
    {relation_algebra_catalog_schema_version_v1,
        sizeof(relation_algebra_catalog_entry_v1),
        relation_algebra_kind_v1::contract_on_support,
        operation_core_compatibility_v1::requires_schema_v2,
        {}, relation_algebra_operation_core_schema_v2,
        candidate_identity(relation_algebra_kind_v1::contract_on_support), {}},
    {relation_algebra_catalog_schema_version_v1,
        sizeof(relation_algebra_catalog_entry_v1),
        relation_algebra_kind_v1::segment_reduce,
        operation_core_compatibility_v1::requires_schema_v2,
        {}, relation_algebra_operation_core_schema_v2,
        candidate_identity(relation_algebra_kind_v1::segment_reduce), {}},
    {relation_algebra_catalog_schema_version_v1,
        sizeof(relation_algebra_catalog_entry_v1),
        relation_algebra_kind_v1::segment_normalize,
        operation_core_compatibility_v1::requires_schema_v2,
        {}, relation_algebra_operation_core_schema_v2,
        candidate_identity(relation_algebra_kind_v1::segment_normalize), {}},
    {relation_algebra_catalog_schema_version_v1,
        sizeof(relation_algebra_catalog_entry_v1),
        relation_algebra_kind_v1::edge_map_or_gate,
        operation_core_compatibility_v1::requires_schema_v2,
        {}, relation_algebra_operation_core_schema_v2,
        candidate_identity(relation_algebra_kind_v1::edge_map_or_gate), {}},
    {relation_algebra_catalog_schema_version_v1,
        sizeof(relation_algebra_catalog_entry_v1),
        relation_algebra_kind_v1::relation_bundle_apply,
        operation_core_compatibility_v1::requires_schema_v2,
        {}, relation_algebra_operation_core_schema_v2,
        candidate_identity(relation_algebra_kind_v1::relation_bundle_apply), {}}};

core::operation_status invalid_catalog(const char *message) noexcept {
    return {core::operation_status_code::invalid_argument,
        execution::binding_validation_code::ok, message};
}

} // namespace

relation_algebra_catalog_view_v1 relation_algebra_candidate_catalog_v1() noexcept {
    return {entries, relation_algebra_catalog_entry_count_v1,
        fragments, relation_algebra_catalog_fragment_count_v1};
}

const relation_algebra_catalog_entry_v1 *find_relation_algebra_catalog_entry_v1(
    relation_algebra_kind_v1 kind) noexcept {
    for (const relation_algebra_catalog_entry_v1 &entry : entries)
        if (entry.relation_kind == kind) return &entry;
    return nullptr;
}

const core::candidate_descriptor_v2 *find_relation_algebra_candidate_v2(
    relation_algebra_kind_v1 kind) noexcept {
    const relation_algebra_catalog_entry_v1 *entry =
        find_relation_algebra_catalog_entry_v1(kind);
    if (entry == nullptr) return nullptr;
    for (const core::candidate_catalog_fragment_v2 &fragment : fragments)
        for (std::uint32_t index = 0u; index < fragment.entry_count; ++index)
            if (core::same_stable_id(
                    fragment.entries[index].candidate.identity,
                    entry->candidate_identity))
                return &fragment.entries[index];
    return nullptr;
}

core::operation_status validate_relation_algebra_candidate_catalog_v1() noexcept {
    for (const core::candidate_catalog_fragment_v2 &fragment : fragments)
        if (core::validate_candidate_catalog_fragment_v2(fragment)
            != core::candidate_catalog_status_v2::success)
            return invalid_catalog("relation candidate fragment is invalid");

    for (std::uint32_t index = 0u;
         index < relation_algebra_catalog_entry_count_v1; ++index) {
        const relation_algebra_catalog_entry_v1 &entry = entries[index];
        if (entry.schema_version != relation_algebra_catalog_schema_version_v1
            || entry.record_bytes != sizeof(relation_algebra_catalog_entry_v1))
            return invalid_catalog("relation catalog entry header is invalid");
        for (std::uint8_t value : entry.reserved0)
            if (value != 0u)
                return invalid_catalog("relation catalog reserved byte is nonzero");
        for (std::uint32_t value : entry.reserved)
            if (value != 0u)
                return invalid_catalog("relation catalog reserved field is nonzero");

        const operation_core_transition_v1 transition =
            operation_core_transition(entry.relation_kind);
        if (entry.compatibility != transition.compatibility
            || entry.required_operation_core_schema
                != (entry.compatibility
                        == operation_core_compatibility_v1::direct_schema_v1
                    ? core::operation_core_schema_version
                    : relation_algebra_operation_core_schema_v2))
            return invalid_catalog("relation catalog schema transition is invalid");

        const core::candidate_descriptor_v2 *candidate =
            find_relation_algebra_candidate_v2(entry.relation_kind);
        if (candidate == nullptr)
            return invalid_catalog("relation catalog candidate is missing");
        if (candidate->projection_contract.variant
            != static_cast<std::uint16_t>(entry.relation_kind))
            return invalid_catalog("relation catalog variant is inconsistent");
        if (entry.compatibility
                == operation_core_compatibility_v1::direct_schema_v1) {
            if (candidate->candidate.operation
                    != core::operation_kind::sparse_dense_multiply
                || (candidate->flags
                    & core::candidate_descriptor_compatibility) == 0u)
                return invalid_catalog("direct relation mapping changed v1 meaning");
        } else {
            if (static_cast<std::uint16_t>(candidate->candidate.operation)
                    != 0x1000u
                        + static_cast<std::uint16_t>(entry.relation_kind)
                || (candidate->flags
                    & core::candidate_descriptor_compatibility) != 0u)
                return invalid_catalog("schema-v2 relation encoding is invalid");
        }
        for (std::uint32_t previous = 0u; previous < index; ++previous)
            if (core::same_stable_id(
                    entry.candidate_identity, entries[previous].candidate_identity))
                return invalid_catalog("relation candidate identity is duplicated");
    }
    return {};
}

} // namespace cellerator::compute::operation
