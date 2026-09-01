#include <Cellerator/profiling/joint_compiler/execution_export_v2.hh>

#include <cassert>
#include <cstdint>

namespace decomposition = cellerator::compute::decomposition;
namespace execution = cellerator::execution;
namespace joint_compiler = cellerator::execution::joint_compiler;
namespace profiling = cellerator::profiling;
namespace export_v2 = cellerator::profiling::joint_compiler;

execution::persistent_axis_identity axis(std::uint64_t seed) {
    return {{execution::biological_abi_version,
                execution::serialized_record_kind::persistent_axis_identity,
                sizeof(execution::persistent_axis_identity)},
        {seed + 1u, 1u}, {seed + 2u, 1u}, {seed + 3u, 1u},
        {seed + 4u, 1u}};
}

void set_numeric(decomposition::decomposition_alternative_v1 *alternative) {
    alternative->numerical.relation_storage = execution::numeric_type::f32;
    alternative->numerical.state_storage = execution::numeric_type::f32;
    alternative->numerical.multiply = execution::numeric_type::f32;
    alternative->numerical.accumulation = execution::numeric_type::f32;
    alternative->numerical.output_storage = execution::numeric_type::f32;
    alternative->numerical.scalar = execution::numeric_type::f32;
}

int main() {
    const std::uint64_t local_to_global[] = {1u, 3u};
    const profiling::export_stage_v1 compatibility_stages[] = {
        {1u, 2u, 1u, 1u}};
    profiling::generic_execution_export_v1 compatibility{};
    compatibility.semantic_geometry_id = 1u;
    compatibility.projection_id = 2u;
    compatibility.candidate_id = 3u;
    compatibility.provider_id = 4u;
    compatibility.capability_id = 5u;
    compatibility.input_order_id = 6u;
    compatibility.output_order_id = 7u;
    compatibility.partition = {8u, 4u, local_to_global, 2u};
    compatibility.stages = compatibility_stages;
    compatibility.stage_count = 1u;

    const joint_compiler::canonical_interval_v1 interval{0u, 2u};
    joint_compiler::logical_coverage_view_v1 coverage{};
    coverage.coverage_identity = {10u, 1u};
    coverage.structure = {11u, 1u};
    coverage.epoch = {1u};
    coverage.source_axis = axis(20u);
    coverage.destination_axis = axis(30u);
    coverage.logical_count = 2u;
    coverage.members = &interval;
    coverage.member_count = 1u;
    coverage.member_bytes = sizeof(interval);

    const joint_compiler::persistent_identity_v1 input_coverages[] = {
        coverage.coverage_identity};
    decomposition::decomposition_alternative_v1 alternative{};
    alternative.alternative_identity = {40u, 1u};
    alternative.candidate_family = {41u, 1u};
    alternative.flags = decomposition::legal_alternative_v1
        | decomposition::complete_unsplit_fallback_v1;
    alternative.required_input_coverages = input_coverages;
    alternative.required_input_coverage_count = 1u;
    alternative.output_coverage = coverage.coverage_identity;
    alternative.input_order = {50u, 1u};
    alternative.output_order = {50u, 2u};
    set_numeric(&alternative);
    decomposition::decomposition_portfolio_v1 portfolio{};
    portfolio.portfolio_identity = {42u, 1u};
    portfolio.alternatives = &alternative;
    portfolio.alternative_count = 1u;

    const joint_compiler::persistent_identity_v1 species[] = {{60u, 1u}};
    const joint_compiler::persistent_identity_v1 required_planes[] = {
        {61u, 1u}};
    joint_compiler::atom_requirement_v1 requirement{};
    requirement.requirement_identity = {62u, 1u};
    requirement.exact_coverage_identity = coverage.coverage_identity;
    requirement.accepted_atom_species = species;
    requirement.accepted_atom_species_count = 1u;
    requirement.required_planes = required_planes;
    requirement.required_plane_count = 1u;
    requirement.numeric = {execution::numeric_type::f32,
        execution::numeric_type::f32, execution::numeric_type::f32, 0u};
    requirement.required_order = alternative.input_order;

    const joint_compiler::atom_plane_affordance_v1 plane = {
        required_planes[0], alternative.input_order, execution::numeric_type::f32,
        execution::numeric_type::f32,
        joint_compiler::mutability_requirement_v1::immutable, 0u, {1u}};
    joint_compiler::atom_affordance_v1 affordance{};
    affordance.affordance_identity = {63u, 1u};
    affordance.atom_species = species[0];
    affordance.exact_coverage_identity = coverage.coverage_identity;
    affordance.physical_encoding = {64u, 1u};
    affordance.local_projection_abi = {65u, 1u};
    affordance.planes = &plane;
    affordance.plane_count = 1u;

    joint_compiler::atom_fragment_result_v1 frontier{};
    frontier.result_identity = {70u, 1u};
    frontier.request_identity = {71u, 1u};
    frontier.no_candidate_reason =
        joint_compiler::no_candidate_reason_v1::unmet_atom_requirement;

    const execution::order_id orders[] = {
        alternative.input_order, alternative.output_order};
    const export_v2::atom_execution_stage_v2 stage = {{80u, 1u}, {81u, 1u},
        coverage.coverage_identity, coverage.coverage_identity, nullptr, 0u, 1u};

    export_v2::execution_export_v2 value{};
    value.export_identity = {90u, 1u};
    value.compatibility_v1 = compatibility;
    value.exact_coverages = &coverage;
    value.exact_coverage_count = 1u;
    value.decomposition = &portfolio;
    value.requirements = &requirement;
    value.requirement_count = 1u;
    value.affordances = &affordance;
    value.affordance_count = 1u;
    value.persistent_orders = orders;
    value.persistent_order_count = 2u;
    value.candidate_frontier = &frontier;
    value.stages = &stage;
    value.stage_count = 1u;
    value.complete_cost.execution_ns = 100u;
    value.complete_cost.expected_reuse = 1u;
    value.correctness =
        export_v2::correctness_compatibility_v2::verified_compatible;
    value.correctness_receipt = {91u, 1u};
    value.performance.status = export_v2::performance_freshness_v2::current;
    value.performance.evidence_identity = {92u, 1u};
    value.performance.device_performance_identity = {93u, 1u};
    value.performance.build_identity = {94u, 1u};
    value.performance.evidence_revision = 1u;
    assert(export_v2::validate_execution_export_v2(value));

    auto malformed = value;
    malformed.compatibility_v1.provider_id = 0u;
    assert(export_v2::validate_execution_export_v2(malformed).code
        == export_v2::execution_export_validation_code_v2::
            invalid_v1_compatibility);

    const std::uint32_t impossible_dependency[] = {0u};
    auto malformed_stage = stage;
    malformed_stage.dependencies = impossible_dependency;
    malformed_stage.dependency_count = 1u;
    malformed = value;
    malformed.stages = &malformed_stage;
    assert(export_v2::validate_execution_export_v2(malformed).code
        == export_v2::execution_export_validation_code_v2::
            invalid_stage_dependency);

    malformed = value;
    malformed.correctness_receipt = {};
    assert(export_v2::validate_execution_export_v2(malformed).code
        == export_v2::execution_export_validation_code_v2::
            invalid_correctness_receipt);

    malformed = value;
    malformed.performance.status =
        export_v2::performance_freshness_v2::analytical_only;
    assert(export_v2::validate_execution_export_v2(malformed).code
        == export_v2::execution_export_validation_code_v2::
            invalid_performance_freshness);
    malformed.performance = {};
    assert(export_v2::validate_execution_export_v2(malformed));

    malformed = value;
    malformed.schema_version = 1u;
    assert(export_v2::validate_execution_export_v2(malformed).code
        == export_v2::execution_export_validation_code_v2::unsupported_schema);
    return 0;
}
