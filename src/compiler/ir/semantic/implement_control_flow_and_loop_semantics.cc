#include <Cellerator/compiler/ir/semantic/implement_control_flow_and_loop_semantics_v1.hh>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace Cellerator::compiler::ir::semantic {

control_flow_status_ir_v1 validate_control_regions_ir_v1(
    const std::vector<control_region_ir_v1>& regions) noexcept {
    std::unordered_set<std::uint64_t> identities;
    for (const auto& region : regions) {
        if (region.identity == 0 || !identities.insert(region.identity).second)
            return control_flow_status_ir_v1::invalid_region;
    }
    for (const auto& region : regions) {
        if (std::any_of(region.child_regions.begin(), region.child_regions.end(),
                        [&](std::uint64_t child) {
                            return child == region.identity || identities.count(child) == 0;
                        })) return control_flow_status_ir_v1::invalid_structure;
        if (region.kind == control_region_kind_ir_v1::branch && region.child_regions.size() != 2)
            return control_flow_status_ir_v1::invalid_structure;
        if (region.kind == control_region_kind_ir_v1::loop &&
            (region.child_regions.size() != 1 || region.bounded_trip_count == 0))
            return control_flow_status_ir_v1::invalid_structure;
        if (!region.semantic_extraction_available &&
            (region.kind != control_region_kind_ir_v1::opaque_cxx_control ||
             (region.effects & control_effect_opaque_barrier_v1) == 0))
            return control_flow_status_ir_v1::invalid_structure;
    }
    return control_flow_status_ir_v1::success;
}

control_flow_status_ir_v1 join_control_dataflow_ir_v1(
    const control_dataflow_state_ir_v1& left,
    const control_dataflow_state_ir_v1& right,
    std::size_t maximum_profile_alternatives,
    control_dataflow_state_ir_v1* result) noexcept {
    if (result == nullptr || maximum_profile_alternatives == 0)
        return control_flow_status_ir_v1::invalid_dataflow;
    control_dataflow_state_ir_v1 joined;
    joined.profiles = left.profiles;
    for (const auto& alternative : right.profiles) {
        if (!alternative.profile.valid() || alternative.probability < 0.0 ||
            alternative.probability > 1.0)
            return control_flow_status_ir_v1::invalid_profile;
        const auto found = std::find_if(joined.profiles.begin(), joined.profiles.end(),
            [&](const profile_alternative_ir_v1& current) {
                return current.profile.low == alternative.profile.low &&
                    current.profile.high == alternative.profile.high;
            });
        if (found == joined.profiles.end()) joined.profiles.push_back(alternative);
        else found->probability = std::max(found->probability, alternative.probability);
    }
    if (joined.profiles.size() > maximum_profile_alternatives)
        return control_flow_status_ir_v1::profile_alternative_limit;
    joined.values = left.values;
    for (const auto& incoming : right.values) {
        if (!incoming.value.valid()) return control_flow_status_ir_v1::invalid_dataflow;
        const auto found = std::find_if(joined.values.begin(), joined.values.end(),
            [&](const control_value_state_ir_v1& current) {
                return current.value.low == incoming.value.low &&
                    current.value.high == incoming.value.high;
            });
        if (found == joined.values.end()) joined.values.push_back(incoming);
        else {
            if (found->generation != incoming.generation) found->generation = 0;
            found->effects |= incoming.effects;
        }
    }
    joined.effects = left.effects | right.effects;
    *result = std::move(joined);
    return control_flow_status_ir_v1::success;
}

}  // namespace Cellerator::compiler::ir::semantic
