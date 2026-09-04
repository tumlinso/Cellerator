#include "cmake/compiler/part_one_acceptance_v1.hh"

#include <cassert>
#include <set>
#include <string_view>

int main() {
    using namespace cellerator::compiler::acceptance::v1;
    assert(part_one_completion.interface_id == "CE-CCP1-I41-PART1-COMPLETE");
    assert(part_one_completion.checkpoint_id == "CE-CCP1-CP-J03");
    assert(part_one_completion.rendezvous_id == "CE-CCP1-RV-M90");
    assert(part_one_completion.integration_task_id == "CE-CCP1-M90");
    assert(part_one_completion.host_sdk_validated);
    assert(part_one_completion.nvidia_sdk_validated);
    assert(part_one_completion.jbc_preserved);
    assert(part_one_completion.part_two_deferred);

    const std::set<std::string_view> inputs(completion_inputs.begin(), completion_inputs.end());
    assert(inputs.size() == completion_inputs.size());
    assert(inputs.count("CE-CCP1-I38-PACKAGE") == 1);
    assert(inputs.count("CE-CCP1-I40-CELLERATORD-SEMANTIC") == 1);
    assert(inputs.count("CE-CCP1-I34-CELLERATOR-LTO") == 1);
    assert(inputs.count("CE-CCP1-I32-DIAGNOSTICS-PROVENANCE") == 1);
}
