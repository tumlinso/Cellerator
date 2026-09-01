#include <Cellerator/compute/decomposition/provider_registry_v1.hh>

#include <Cellerator/compute/decomposition/dense_width_v1.hh>
#include <Cellerator/compute/decomposition/destination_disjoint_v1.hh>
#include <Cellerator/compute/decomposition/edge_component_v1.hh>
#include <Cellerator/compute/decomposition/relation_bundle_v1.hh>
#include <Cellerator/compute/decomposition/segment_disjoint_v1.hh>
#include <Cellerator/compute/decomposition/source_k_v1.hh>
#include <Cellerator/compute/decomposition/split_segment_reduce_v1.hh>
#include <Cellerator/compute/decomposition/support_contraction_v1.hh>
#include <Cellerator/compute/decomposition/support_edge_rectangle_v1.hh>
#include <Cellerator/compute/decomposition/support_embedding_v1.hh>
#include <Cellerator/compute/decomposition/transpose_source_partials_v1.hh>

namespace cellerator::compute::decomposition {
namespace {

template<typename Instance, typename Validator>
provider_instance_validation_code_v1 validate_without_workspace(
    const void *instance, Validator validator) noexcept {
    if (instance == nullptr)
        return provider_instance_validation_code_v1::missing_instance;
    return validator(*static_cast<const Instance *>(instance))
        ? provider_instance_validation_code_v1::ok
        : provider_instance_validation_code_v1::invalid_instance;
}

provider_instance_validation_code_v1 validate_destination(
    const void *instance, provider_validation_workspace_v1) noexcept {
    return validate_without_workspace<destination_disjoint_relation_apply_v1>(
        instance, validate_destination_disjoint_relation_apply_v1);
}

provider_instance_validation_code_v1 validate_source_k(const void *instance,
    provider_validation_workspace_v1) noexcept {
    return validate_without_workspace<source_k_relation_apply_v1>(
        instance, validate_source_k_relation_apply_v1);
}

provider_instance_validation_code_v1 validate_dense(const void *instance,
    provider_validation_workspace_v1) noexcept {
    return validate_without_workspace<dense_width_relation_apply_v1>(
        instance, validate_dense_width_relation_apply_v1);
}

provider_instance_validation_code_v1 validate_edge_component(
    const void *instance, provider_validation_workspace_v1 workspace) noexcept {
    if (instance == nullptr)
        return provider_instance_validation_code_v1::missing_instance;
    const auto &value =
        *static_cast<const edge_component_relation_apply_v1 *>(instance);
    if (workspace.data == nullptr || value.cover == nullptr
        || workspace.byte_count < value.cover->logical_edge_count)
        return provider_instance_validation_code_v1::missing_workspace;
    geometry::relation_cover_validation_workspace cover_workspace{
        static_cast<std::uint8_t *>(workspace.data), workspace.byte_count};
    return validate_edge_component_relation_apply_v1(value, cover_workspace)
        ? provider_instance_validation_code_v1::ok
        : provider_instance_validation_code_v1::invalid_instance;
}

provider_instance_validation_code_v1 validate_bundle(const void *instance,
    provider_validation_workspace_v1) noexcept {
    return validate_without_workspace<relation_bundle_type_decomposition_v1>(
        instance, validate_relation_bundle_type_decomposition_v1);
}

provider_instance_validation_code_v1 validate_transpose(const void *instance,
    provider_validation_workspace_v1) noexcept {
    return validate_without_workspace<transpose_source_partials_v1>(
        instance, validate_transpose_source_partials_v1);
}

provider_instance_validation_code_v1 validate_support_axis(
    const void *instance, provider_validation_workspace_v1) noexcept {
    return validate_without_workspace<support_contraction_decomposition_v1>(
        instance, validate_support_contraction_decomposition_v1);
}

provider_instance_validation_code_v1 validate_support_edges(
    const void *instance, provider_validation_workspace_v1 workspace) noexcept {
    if (instance == nullptr)
        return provider_instance_validation_code_v1::missing_instance;
    const auto &value = *static_cast<
        const support_edge_rectangle_decomposition_v1 *>(instance);
    if (workspace.data == nullptr || value.cover == nullptr
        || workspace.byte_count < value.cover->logical_edge_count)
        return provider_instance_validation_code_v1::missing_workspace;
    geometry::relation_cover_validation_workspace cover_workspace{
        static_cast<std::uint8_t *>(workspace.data), workspace.byte_count};
    return validate_support_edge_rectangle_decomposition_v1(
            value, cover_workspace)
        ? provider_instance_validation_code_v1::ok
        : provider_instance_validation_code_v1::invalid_instance;
}

provider_instance_validation_code_v1 validate_support_embedding(
    const void *instance, provider_validation_workspace_v1) noexcept {
    return validate_without_workspace<support_embedding_decomposition_v1>(
        instance, validate_support_embedding_decomposition_v1);
}

provider_instance_validation_code_v1 validate_segments(const void *instance,
    provider_validation_workspace_v1) noexcept {
    return validate_without_workspace<segment_disjoint_decomposition_v1>(
        instance, validate_segment_disjoint_v1);
}

provider_instance_validation_code_v1 validate_split_segments(
    const void *instance, provider_validation_workspace_v1) noexcept {
    return validate_without_workspace<split_segment_reduce_decomposition_v1>(
        instance, validate_split_segment_reduce_v1);
}

constexpr decomposition_provider_v1 provider(std::uint64_t low,
    decomposition_provider_kind_v1 kind,
    operation::v2::operation_kind operation,
    split_axis_kind_v1 axis,
    provider_partial_mode_v1 partial_mode,
    provider_validate_instance_fn_v1 validator) noexcept {
    return {{low, 1u}, {low, 2u}, kind, operation, axis, partial_mode,
        true, true, 0u, 1u, validator};
}

const decomposition_provider_v1 builtin_providers[] = {
    provider(1u, decomposition_provider_kind_v1::destination_disjoint,
        operation::v2::operation_kind::relation_apply,
        split_axis_kind_v1::destination, provider_partial_mode_v1::never,
        validate_destination),
    provider(2u, decomposition_provider_kind_v1::source_k,
        operation::v2::operation_kind::relation_apply,
        split_axis_kind_v1::source, provider_partial_mode_v1::always,
        validate_source_k),
    provider(3u, decomposition_provider_kind_v1::dense_width,
        operation::v2::operation_kind::relation_apply,
        split_axis_kind_v1::dense_channel, provider_partial_mode_v1::never,
        validate_dense),
    provider(4u, decomposition_provider_kind_v1::edge_component,
        operation::v2::operation_kind::relation_apply,
        split_axis_kind_v1::semantic_component,
        provider_partial_mode_v1::always, validate_edge_component),
    provider(5u, decomposition_provider_kind_v1::relation_bundle_type,
        operation::v2::operation_kind::relation_bundle_apply,
        split_axis_kind_v1::none, provider_partial_mode_v1::always,
        validate_bundle),
    provider(6u, decomposition_provider_kind_v1::transpose_source_partial,
        operation::v2::operation_kind::relation_apply_transpose,
        split_axis_kind_v1::destination, provider_partial_mode_v1::always,
        validate_transpose),
    provider(7u, decomposition_provider_kind_v1::support_axis,
        operation::v2::operation_kind::contract_on_support,
        split_axis_kind_v1::none, provider_partial_mode_v1::instance_dependent,
        validate_support_axis),
    provider(8u, decomposition_provider_kind_v1::support_edge_rectangle,
        operation::v2::operation_kind::contract_on_support,
        split_axis_kind_v1::none, provider_partial_mode_v1::always,
        validate_support_edges),
    provider(9u, decomposition_provider_kind_v1::support_embedding,
        operation::v2::operation_kind::contract_on_support,
        split_axis_kind_v1::dense_channel, provider_partial_mode_v1::never,
        validate_support_embedding),
    provider(10u, decomposition_provider_kind_v1::segment_disjoint,
        operation::v2::operation_kind::segment_reduce,
        split_axis_kind_v1::none, provider_partial_mode_v1::never,
        validate_segments),
    provider(11u, decomposition_provider_kind_v1::split_segment_reduce,
        operation::v2::operation_kind::segment_reduce,
        split_axis_kind_v1::logical_edge, provider_partial_mode_v1::always,
        validate_split_segments)};

bool valid_kind(decomposition_provider_kind_v1 kind) noexcept {
    return kind >= decomposition_provider_kind_v1::destination_disjoint
        && kind <= decomposition_provider_kind_v1::split_segment_reduce;
}

bool ordered_after(operation::v2::stable_id current,
    operation::v2::stable_id previous) noexcept {
    return current.high > previous.high
        || (current.high == previous.high && current.low > previous.low);
}

}  // namespace

decomposition_provider_registry_v1 builtin_decomposition_providers_v1() noexcept {
    return {provider_registry_schema_version_v1, 0u, builtin_providers,
        builtin_provider_count_v1};
}

provider_registry_validation_result_v1 validate_provider_registry_v1(
    const decomposition_provider_registry_v1 &registry) noexcept {
    using code = provider_registry_validation_code_v1;
    if (registry.schema_version != provider_registry_schema_version_v1)
        return {code::unsupported_schema, 0u};
    if (registry.reserved != 0u)
        return {code::nonzero_reserved, 0u};
    if (registry.provider_count == 0u)
        return {code::invalid_provider_count, 0u};
    if (registry.providers == nullptr)
        return {code::missing_providers, 0u};
    operation::v2::stable_id previous{};
    for (std::uint64_t index = 0u; index < registry.provider_count; ++index) {
        const auto &entry = registry.providers[index];
        if (!operation::v2::valid_stable_id(entry.provider_identity)
            || !operation::v2::valid_stable_id(
                entry.independent_validation_identity))
            return {code::invalid_provider_identity, index};
        if (operation::v2::same_stable_id(entry.provider_identity,
                entry.independent_validation_identity))
            return {code::validation_identity_alias, index};
        if (index != 0u && !ordered_after(entry.provider_identity, previous))
            return {code::provider_order_mismatch, index};
        previous = entry.provider_identity;
        if (!valid_kind(entry.kind))
            return {code::invalid_kind, index};
        if (!operation::v2::valid_operation_kind(entry.operation))
            return {code::invalid_operation, index};
        if (!valid_split_axis_kind_v1(entry.primary_split_axis))
            return {code::invalid_split_axis, index};
        if (entry.partial_mode < provider_partial_mode_v1::never
            || entry.partial_mode > provider_partial_mode_v1::instance_dependent)
            return {code::invalid_partial_mode, index};
        if (!entry.unsplit_fallback_available)
            return {code::missing_unsplit_fallback, index};
        if (!entry.requires_exact_coverage)
            return {code::missing_exact_coverage, index};
        if (entry.reserved != 0u)
            return {code::nonzero_reserved, index};
        if (entry.validation_revision == 0u)
            return {code::missing_validation_revision, index};
        if (entry.validate_instance == nullptr)
            return {code::missing_validator, index};
    }
    return {};
}

provider_lookup_result_v1 find_decomposition_provider_v1(
    const decomposition_provider_registry_v1 &registry,
    decomposition_provider_kind_v1 kind) noexcept {
    if (!validate_provider_registry_v1(registry))
        return {nullptr, provider_lookup_code_v1::invalid_registry};
    for (std::uint64_t index = 0u; index < registry.provider_count; ++index) {
        if (registry.providers[index].kind == kind)
            return {registry.providers + index, provider_lookup_code_v1::found};
    }
    return {nullptr, provider_lookup_code_v1::no_candidate};
}

}  // namespace cellerator::compute::decomposition
