#include <Cellerator/compute/operation/relation_algebra.hh>

#include <array>
#include <cassert>
#include <cstdint>

namespace operation = cellerator::compute::operation;
namespace execution = cellerator::execution;

namespace {

execution::persistent_axis_identity axis(std::uint64_t seed) {
    execution::persistent_axis_identity result{};
    result.header = {execution::biological_abi_version,
        execution::serialized_record_kind::persistent_axis_identity,
        sizeof(result)};
    result.domain = {seed + 1u, seed + 2u};
    result.order = {seed + 3u, seed + 4u};
    result.geometry = {seed + 5u, seed + 6u};
    result.partition = {seed + 7u, seed + 8u};
    return result;
}

operation::relation_numeric_semantics_v1 fp32_numeric() {
    operation::relation_numeric_semantics_v1 result{};
    result.relation_storage = execution::numeric_type::f32;
    result.state_storage = execution::numeric_type::f32;
    result.multiply = execution::numeric_type::f32;
    result.accumulation = execution::numeric_type::f32;
    result.output_storage = execution::numeric_type::f32;
    result.scalar = execution::numeric_type::f32;
    return result;
}

operation::typed_relation_v1 relation(std::uint64_t seed,
    execution::persistent_axis_identity source,
    execution::persistent_axis_identity destination) {
    operation::typed_relation_v1 result{};
    result.structure = {seed + 1u, seed + 2u};
    result.epoch = {1u};
    result.source_axis = source;
    result.destination_axis = destination;
    result.logical_edge_count = 16u;
    return result;
}

operation::relation_algebra_problem_v1 apply_problem(std::uint64_t seed,
    operation::relation_algebra_kind_v1 kind,
    operation::typed_relation_v1 typed) {
    operation::relation_algebra_problem_v1 result{};
    result.kind = kind;
    result.operation_identity = {seed + 1u, seed + 2u};
    result.relation = typed;
    result.numeric = fp32_numeric();
    result.semantic_flags = operation::alpha_applied_once
        | operation::beta_applied_once;
    result.dense_width = 8u;
    return result;
}

operation::relation_algebra_problem_v1 sparse_state_embedding() {
    const auto genes = axis(0x100u);
    const auto cells = axis(0x200u);
    return apply_problem(0x1000u,
        operation::relation_algebra_kind_v1::relation_apply,
        relation(0x1100u, genes, cells));
}

operation::relation_algebra_problem_v1 regulatory_propagation() {
    const auto regulators = axis(0x300u);
    const auto genes = axis(0x400u);
    return apply_problem(0x2000u,
        operation::relation_algebra_kind_v1::relation_apply,
        relation(0x2100u, regulators, genes));
}

operation::relation_algebra_problem_v1 transition_transport() {
    const auto source_cells = axis(0x500u);
    const auto destination_cells = axis(0x600u);
    return apply_problem(0x3000u,
        operation::relation_algebra_kind_v1::relation_apply,
        relation(0x3100u, source_cells, destination_cells));
}

operation::relation_algebra_problem_v1 hierarchy_incidence() {
    const auto children = axis(0x700u);
    const auto parents = axis(0x800u);
    return apply_problem(0x4000u,
        operation::relation_algebra_kind_v1::relation_apply_transpose,
        relation(0x4100u, parents, children));
}

operation::relation_algebra_problem_v1 multimodal_relations(
    std::array<operation::typed_relation_v1, 2> *storage) {
    const auto genes = axis(0x900u);
    const auto peaks = axis(0xa00u);
    const auto cells = axis(0xb00u);
    (*storage)[0] = relation(0x5100u, genes, cells);
    (*storage)[1] = relation(0x5200u, peaks, cells);
    operation::relation_algebra_problem_v1 result{};
    result.kind = operation::relation_algebra_kind_v1::relation_bundle_apply;
    result.operation_identity = {0x5001u, 0x5002u};
    result.bundle = {storage->data(), static_cast<std::uint32_t>(storage->size()),
        0u, cells};
    result.numeric = fp32_numeric();
    result.semantic_flags = operation::sequential_bundle_is_valid;
    return result;
}

operation::relation_algebra_problem_v1 perturbation_delta_propagation() {
    const auto perturbations = axis(0xc00u);
    const auto genes = axis(0xd00u);
    return apply_problem(0x6000u,
        operation::relation_algebra_kind_v1::relation_apply,
        relation(0x6100u, perturbations, genes));
}

} // namespace

int main() {
    std::array<operation::typed_relation_v1, 2> multimodal_storage{};
    const operation::relation_algebra_problem_v1 examples[] = {
        sparse_state_embedding(),
        regulatory_propagation(),
        transition_transport(),
        hierarchy_incidence(),
        multimodal_relations(&multimodal_storage),
        perturbation_delta_propagation(),
    };
    for (const auto &example : examples)
        assert(operation::validate_relation_algebra_problem_v1(example)
            == operation::relation_algebra_status_v1::ok);
    return 0;
}
