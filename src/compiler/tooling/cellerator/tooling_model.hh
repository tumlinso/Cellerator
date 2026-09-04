#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace cellerator::compiler::tooling::v1 {

struct completion_item {
    std::string spelling;
    std::string category;
};

[[nodiscard]] std::vector<completion_item>
complete_cellerator_syntax(std::string_view source, std::size_t cursor);

struct biological_hover {
    std::string domain, tag, source_axis, destination_axis, support, orientation;
    std::string numeric_tuple, mutability, structure_identity, value_generation, source_link;
};
[[nodiscard]] biological_hover describe_biological_relation(std::string_view declaration);
struct field_effect_view {std::string field,boundary,profile;std::vector<std::string> captures,reads,writes,effects,barriers;bool optimization_visible=false;};
[[nodiscard]] field_effect_view describe_field_effects(std::string_view source,std::size_t cursor);
struct profile_state_view {std::string selected;std::vector<std::string> evidence,alternatives,unknown_dimensions,missing_hints;double confidence=0;std::string support_state,value_state,mutation_state;};
[[nodiscard]] profile_state_view profile_state_at_cursor(std::string_view source,std::size_t cursor);
struct generation_view {std::size_t structure=1,value=1,support=1,order=1;std::vector<std::string> stale_artifacts;};
[[nodiscard]] generation_view query_generations(std::string_view statement);
struct semantic_ir_view {std::string normalized,source_map;std::vector<std::string> effects,profiles,extensions;};
[[nodiscard]] semantic_ir_view semantic_ir_at_cursor(std::string_view source,std::size_t cursor);
[[nodiscard]] std::string apply_semantic_ir_edit(std::string_view source,const semantic_ir_view& edit);
struct candidate_view {std::string name,decomposition,evidence,rejected_reason;double cost=0;std::size_t resources=0;bool certified=false,selected=false,forced=false;};
struct planning_ir_view {std::string problem,exact_cover;std::vector<std::string> atom_proposals;std::vector<candidate_view> candidates;};
[[nodiscard]] planning_ir_view planning_ir_at_cursor(bool force_reference);
struct realization_view {std::vector<std::string> atoms,extents,projections,orders,partial_tree,stages,dependencies;std::size_t workspace_bytes=0;std::string target,readiness;bool graph_capture=false;};
[[nodiscard]] realization_view realization_at_cursor();
[[nodiscard]] std::string render_realization_json(const realization_view&);
struct candidate_explanation {double complete_cost=0,transition_cost=0;std::string evidence_kind,freshness,uncertainty,constraints,reuse,alternatives,dominance,user_edits,fallback;};
[[nodiscard]] candidate_explanation explain_candidate(std::string_view evidence_kind,bool forced);
struct optimization_action {std::string cause,detail,fix_it;double canonicalization_cost=0;bool safe=false;};
[[nodiscard]] std::vector<optimization_action> missed_optimization_actions(std::string_view source);

}  // namespace cellerator::compiler::tooling::v1
