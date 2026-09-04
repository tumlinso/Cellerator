#include <Cellerator/compiler/pass/freeze_the_pass_pipeline_stage_taxonomy_v1.hh>

#include <array>

namespace cellerator::compiler::pass::v1 {

std::string_view pipeline_phase_name_v1(pipeline_phase_v1 phase) noexcept {
    static constexpr std::array<std::string_view, pipeline_phase_count_v1> names{
        "source-canonicalization", "profile-propagation", "discovery",
        "certification", "decomposition", "candidate-enumeration",
        "cost-modeling", "selection", "realization", "packing",
        "stage-construction", "backend-emission"};
    const auto index = static_cast<std::size_t>(phase);
    return index < names.size() ? names[index] : std::string_view{};
}

ir::realization::v1::ceir_facet_v1 lowering_resumption_facet_v1(
    pipeline_phase_v1 phase) noexcept {
    using facet = ir::realization::v1::ceir_facet_v1;
    switch (phase) {
    case pipeline_phase_v1::source_canonicalization: return facet::canonical_source;
    case pipeline_phase_v1::profile_propagation:
    case pipeline_phase_v1::discovery: return facet::atom_evidence;
    case pipeline_phase_v1::certification: return facet::semantic_atom;
    case pipeline_phase_v1::decomposition:
    case pipeline_phase_v1::candidate_enumeration:
    case pipeline_phase_v1::cost_modeling: return facet::target_cover;
    case pipeline_phase_v1::selection: return facet::physical_projection;
    case pipeline_phase_v1::realization: return facet::local_realization;
    case pipeline_phase_v1::packing: return facet::packed_operand;
    case pipeline_phase_v1::stage_construction: return facet::executable_recipe;
    case pipeline_phase_v1::backend_emission: return facet::local_realization;
    case pipeline_phase_v1::count: return facet::count;
    }
    return facet::count;
}

bool valid_pipeline_stage_v1(pipeline_stage_v1 stage) noexcept {
    return static_cast<std::size_t>(stage.phase) < pipeline_phase_count_v1
        && (stage.side == interception_side_v1::before
            || stage.side == interception_side_v1::after);
}

}  // namespace cellerator::compiler::pass::v1
