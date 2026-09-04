#include <Cellerator/compiler/ir/planning/freeze_planning_ir_module_and_decision_state_model_v1.hh>

#include <array>
#include <cassert>
#include <cstddef>

int main() {
    using namespace cellerator::compiler::ir::planning::v1;
    constexpr std::array<decision_state_v1, 9> states{
        decision_state_v1::unresolved, decision_state_v1::offered,
        decision_state_v1::admissible, decision_state_v1::rejected,
        decision_state_v1::dominated, decision_state_v1::selected,
        decision_state_v1::forced, decision_state_v1::externally_selected,
        decision_state_v1::fallback};
    std::array<decision_record_v1, states.size()> records{};
    for (std::size_t index = 0u; index != records.size(); ++index) {
        records[index].decision = {index + 1u, 100u + index};
        records[index].candidate = {200u + index, 300u + index};
        records[index].source_operation = {400u, 500u};
        records[index].state = states[index];
        records[index].flags = index < 5u ? decision_flag_none_v1 : decision_flag_correct_v1;
        records[index].evidence_revision = 600u + index;
    }
    const planning_ir_module_v1 input{planning_ir_schema_version_v1, 0u, {9u, 10u},
                                      records.data(), records.size(), 0u};
    assert(validate_planning_ir_module_v1(input) == planning_ir_status_v1::ok);

    std::array<std::byte, 2048> storage{};
    std::size_t written = 0u;
    assert(serialize_planning_decisions_v1(input, storage.data(), storage.size(), &written) ==
           planning_ir_status_v1::ok);
    std::array<decision_record_v1, states.size()> decoded{};
    planning_ir_module_v1 output{};
    assert(deserialize_planning_decisions_v1(storage.data(), written, decoded.data(),
                                             decoded.size(), &output) == planning_ir_status_v1::ok);
    assert(output.module.low == input.module.low && output.decision_count == input.decision_count);
    for (std::size_t index = 0u; index != records.size(); ++index) {
        assert(decoded[index].decision.low == records[index].decision.low);
        assert(decoded[index].candidate.high == records[index].candidate.high);
        assert(decoded[index].state == records[index].state);
        assert(decoded[index].flags == records[index].flags);
        assert(decoded[index].evidence_revision == records[index].evidence_revision);
    }

    auto duplicate = records;
    duplicate[1].decision = duplicate[0].decision;
    auto invalid = input;
    invalid.decisions = duplicate.data();
    assert(validate_planning_ir_module_v1(invalid) ==
           planning_ir_status_v1::duplicate_decision);
    assert(deserialize_planning_decisions_v1(storage.data(), written - 1u, decoded.data(),
                                             decoded.size(), &output) ==
           planning_ir_status_v1::malformed_binary);
}
