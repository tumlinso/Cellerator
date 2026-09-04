#include <Cellerator/compiler/planning/implement_custom_candidate_registration_v1.hh>

#include <cassert>

namespace planning = Cellerator::compiler::planning;

int main() {
    planning::custom_candidate_registry_v1 registry;
    planning::custom_candidate_registration_v1 external{};
    external.candidate_identity = 101u;
    external.provider_identity = 102u;
    external.operation_identity = 103u;
    external.origin = planning::custom_candidate_origin_v1::external_library;
    external.provided_protocols = planning::custom_protocol_execute_v1;
    external.required_protocols = planning::custom_protocol_prepare_v1 |
        planning::custom_protocol_execute_v1 | planning::custom_protocol_profile_v1;
    external.missing_behavior = planning::missing_protocol_behavior_v1::opaque_passthrough;
    external.stable_name = "external.fast-path";
    external.source_locator = "libexternal_candidate.so";
    external.entry_symbol = "register_fast_path_v1";
    external.opaque_payload = {0xdeu, 0xadu, 0xbeu, 0xefu};
    assert(planning::register_custom_candidate_v1(&registry, external) ==
        planning::custom_candidate_registration_code_v1::ok);

    planning::custom_candidate_registry_v1 binary_roundtrip;
    const auto binary = planning::write_custom_candidate_binary_ir_v1(registry);
    assert(planning::read_custom_candidate_binary_ir_v1(binary, &binary_roundtrip) ==
        planning::custom_candidate_registration_code_v1::ok);
    assert(planning::equivalent_custom_candidate_registries_v1(
        registry, binary_roundtrip));

    planning::custom_candidate_registry_v1 text_roundtrip;
    const auto text = planning::write_custom_candidate_text_ir_v1(registry);
    assert(planning::read_custom_candidate_text_ir_v1(text, &text_roundtrip) ==
        planning::custom_candidate_registration_code_v1::ok);
    assert(planning::equivalent_custom_candidate_registries_v1(registry, text_roundtrip));
    assert(text_roundtrip.candidates[0].missing_behavior ==
        planning::missing_protocol_behavior_v1::opaque_passthrough);

    external.candidate_identity = 201u;
    external.missing_behavior = planning::missing_protocol_behavior_v1::reject;
    assert(planning::register_custom_candidate_v1(&registry, external) ==
        planning::custom_candidate_registration_code_v1::incomplete_protocol);
}
