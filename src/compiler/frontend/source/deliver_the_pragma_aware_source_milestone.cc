#include <Cellerator/compiler/frontend/source/deliver_the_pragma_aware_source_milestone_v1.hh>

#include <Cellerator/compiler/frontend/source/classify_source_inputs_independently_of_extension_v1.hh>
#include <Cellerator/compiler/frontend/source/recognize_cellerator_execution_field_token_islands_v1.hh>

namespace Cellerator::compiler::frontend::source {

std::vector<transformed_source_unit_v1> transform_pragma_aware_sources_v1(
    const std::vector<source_unit_v1>& units) {
    std::vector<transformed_source_unit_v1> result;
    result.reserve(units.size());
    for (const auto& unit : units) {
        const auto classification = classify_source_input_v1(unit.path, unit.bytes);
        if (classification.mode != source_input_mode_v1::activated_cellerator) {
            result.push_back({unit.id, unit.path, unit.bytes, false, {}});
            continue;
        }
        const auto islands = recognize_execution_field_islands_v1(
            unit.id, unit.bytes, classification.activation_offset);
        if (!islands.balanced) {
            result.push_back({unit.id, unit.path, unit.bytes, true, {}});
            continue;
        }
        auto shadow = construct_shadow_cxx_v1(unit.id, unit.bytes, islands.islands);
        result.push_back({unit.id, unit.path, std::move(shadow.bytes), true,
                          std::move(shadow.placeholders)});
    }
    return result;
}

} // namespace Cellerator::compiler::frontend::source
