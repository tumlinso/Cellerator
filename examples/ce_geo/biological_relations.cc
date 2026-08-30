#include <Cellerator/compute/operation/relation_algebra.hh>
#include <Cellerator/execution/lifetimes.hh>

#include <array>
#include <cstdint>
#include <iostream>

namespace {

namespace operation = cellerator::compute::operation;
namespace core = cellerator::compute::math::core;
namespace execution = cellerator::execution;

execution::persistent_axis_identity persistent_axis(
    std::uint64_t seed) noexcept {
    return {
        {execution::biological_abi_version,
         execution::serialized_record_kind::persistent_axis_identity,
         sizeof(execution::persistent_axis_identity)},
        {seed + 1u, seed + 2u},
        {seed + 3u, seed + 4u},
        {seed + 5u, seed + 6u},
        {seed + 7u, seed + 8u}};
}

execution::axis_identity runtime_axis(std::uint32_t seed) noexcept {
    return {{seed + 1u, 1u}, {seed + 2u, 1u},
        {seed + 3u, 1u}, {seed + 4u, 1u}};
}

operation::typed_relation_v1 typed_relation(
    std::uint64_t identity_seed,
    execution::persistent_axis_identity source,
    execution::persistent_axis_identity destination,
    std::uint64_t logical_edge_count) noexcept {
    return {{identity_seed + 1u, identity_seed + 2u},
        {identity_seed + 3u}, source, destination, logical_edge_count};
}

operation::relation_numeric_semantics_v1 numeric_semantics() noexcept {
    return {execution::numeric_type::f32,
        execution::numeric_type::f32,
        execution::numeric_type::f32,
        execution::numeric_type::f32,
        execution::numeric_type::f32,
        execution::numeric_type::f32,
        core::rounding_policy::nearest_even,
        core::saturation_policy::none,
        operation::nan_policy_v1::propagate,
        {}};
}

operation::relation_algebra_problem_v1 relation_apply_problem(
    std::uint64_t operation_seed,
    operation::typed_relation_v1 relation,
    std::uint32_t dense_width,
    operation::relation_algebra_kind_v1 kind =
        operation::relation_algebra_kind_v1::relation_apply) noexcept {
    operation::relation_algebra_problem_v1 problem{};
    problem.kind = kind;
    problem.operation_identity = {operation_seed + 1u, operation_seed + 2u};
    problem.relation = relation;
    problem.numeric = numeric_semantics();
    problem.semantic_flags = operation::alpha_applied_once
        | operation::beta_applied_once;
    problem.dense_width = dense_width;
    return problem;
}

bool sparse_state_embedding() {
    const auto gene_features = persistent_axis(100u);
    const auto latent_state = persistent_axis(200u);
    const auto embedding = typed_relation(
        300u, gene_features, latent_state, 18432u);
    const auto problem = relation_apply_problem(400u, embedding, 32u);
    return operation::validate_relation_algebra_problem_v1(problem)
            == operation::relation_algebra_status_v1::ok
        && operation::operation_core_transition(problem.kind).compatibility
            == operation::operation_core_compatibility_v1::direct_schema_v1;
}

bool regulatory_propagation() {
    // Direction is explicit and mutable edge values are not structure identity.
    const auto regulators = persistent_axis(500u);
    const auto target_genes = persistent_axis(600u);
    const auto regulatory_graph = typed_relation(
        700u, regulators, target_genes, 24000u);
    const auto problem = relation_apply_problem(800u, regulatory_graph, 16u);
    return operation::validate_relation_algebra_problem_v1(problem)
            == operation::relation_algebra_status_v1::ok
        && !operation::same_persistent_axis(regulators, target_genes);
}

bool transition_transport() {
    // Separate persistent orders prevent equal-sized time points from aliasing.
    const auto state_at_t = persistent_axis(900u);
    const auto state_at_t_plus_one = persistent_axis(1000u);
    const auto transition = typed_relation(
        1100u, state_at_t, state_at_t_plus_one, 8192u);
    const auto problem = relation_apply_problem(1200u, transition, 8u);
    return operation::validate_relation_algebra_problem_v1(problem)
            == operation::relation_algebra_status_v1::ok
        && !operation::same_persistent_axis(state_at_t, state_at_t_plus_one);
}

bool hierarchy_incidence() {
    // One incidence structure supports forward pooling and transpose broadcast.
    const auto member_cells = persistent_axis(1300u);
    const auto cell_modules = persistent_axis(1400u);
    const auto incidence = typed_relation(
        1500u, member_cells, cell_modules, 4096u);
    const auto pool = relation_apply_problem(1600u, incidence, 4u);
    const auto broadcast = relation_apply_problem(
        1700u, incidence, 4u,
        operation::relation_algebra_kind_v1::relation_apply_transpose);
    return operation::validate_relation_algebra_problem_v1(pool)
            == operation::relation_algebra_status_v1::ok
        && operation::validate_relation_algebra_problem_v1(broadcast)
            == operation::relation_algebra_status_v1::ok;
}

bool multimodal_relations() {
    // RNA and ATAC axes remain distinct while sharing an exact destination.
    const auto rna_genes = persistent_axis(1800u);
    const auto atac_peaks = persistent_axis(1900u);
    const auto joint_modules = persistent_axis(2000u);
    const operation::typed_relation_v1 relations[] = {
        typed_relation(2100u, rna_genes, joint_modules, 12000u),
        typed_relation(2200u, atac_peaks, joint_modules, 18000u)};
    operation::relation_algebra_problem_v1 problem{};
    problem.kind = operation::relation_algebra_kind_v1::relation_bundle_apply;
    problem.operation_identity = {2301u, 2302u};
    problem.bundle = {relations, 2u, 0u, joint_modules};
    problem.numeric = numeric_semantics();
    problem.semantic_flags = operation::sequential_bundle_is_valid;
    return operation::validate_relation_algebra_problem_v1(problem)
            == operation::relation_algebra_status_v1::ok
        && !operation::same_persistent_axis(rna_genes, atac_peaks);
}

bool perturbation_delta_propagation() {
    // Topology epoch and mutable delta generation are independent identities.
    const auto perturbed_genes = persistent_axis(2400u);
    const auto response_genes = persistent_axis(2500u);
    const auto response = typed_relation(
        2600u, perturbed_genes, response_genes, 4u);
    const auto problem = relation_apply_problem(2700u, response, 1u);
    const execution::relation_structure runtime_structure{
        {1u, 1u}, {response.epoch.value}, runtime_axis(10u), runtime_axis(20u),
        {1u, 1u}, response.logical_edge_count};
    std::array<float, 4> delta_weights{1.0F, -0.5F, 0.25F, 2.0F};
    const execution::value_plane delta_plane{
        runtime_structure.identity, runtime_structure.epoch,
        delta_weights.data(),
        {execution::residency_kind::host, {}, -1, 0u},
        {execution::numeric_type::f32, execution::numeric_type::f32,
         execution::numeric_type::f32, 0u},
        {execution::quantization_kind::none, execution::numeric_type::invalid,
         execution::numeric_type::invalid, 0u, nullptr, nullptr, 0u},
        execution::value_layout_kind::logical_edge_order,
        {}, {7u}, delta_weights.size(), sizeof(delta_weights)};
    const execution::value_binding current_delta{&delta_plane, {7u}};
    const execution::value_binding stale_delta{&delta_plane, {6u}};
    return operation::validate_relation_algebra_problem_v1(problem)
            == operation::relation_algebra_status_v1::ok
        && execution::validate_value_binding(runtime_structure, current_delta)
            == execution::lifetime_validation_code::ok
        && execution::validate_value_binding(runtime_structure, stale_delta)
            == execution::lifetime_validation_code::stale_value_generation;
}

} // namespace

int main() {
    struct scenario {
        const char *name;
        bool (*validate)();
    };
    const scenario scenarios[] = {
        {"sparse state embedding", sparse_state_embedding},
        {"regulatory propagation", regulatory_propagation},
        {"transition/transport", transition_transport},
        {"hierarchy incidence", hierarchy_incidence},
        {"multimodal relations", multimodal_relations},
        {"perturbation delta propagation", perturbation_delta_propagation}};
    for (const scenario &item : scenarios) {
        if (!item.validate()) {
            std::cerr << "biological relation example failed: " << item.name << '\n';
            return 1;
        }
        std::cout << item.name << " contract passed\n";
    }
    return 0;
}
