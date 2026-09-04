#include <Cellerator/compiler/pass/integrate_extension_pass_provenance_v1.hh>

#include <cassert>

namespace cp = cellerator::compiler::pass::v1;

int main() {
    cp::cold_extension_provenance_v1 provenance;
    assert(provenance.append({"org.example.normalize@1", "sha256:source",
        {cp::pipeline_phase_v1::source_canonicalization,
            cp::interception_side_v1::after},
        {10}, {20}, 0x3, {"rewrote source operation"},
        cp::transform_execution_mode_v1::trusted_in_process}));
    assert(provenance.append({"org.example.lower@2", "sha256:binary",
        {cp::pipeline_phase_v1::realization, cp::interception_side_v1::before},
        {20}, {30}, 0x8, {"selected native projection"},
        cp::transform_execution_mode_v1::isolated_verified}));
    const auto trace = provenance.trace_operation(30);
    assert(trace.size() == 2);
    assert(trace[0]->input_operation_ids[0] == 10);
    assert(trace[1]->output_operation_ids[0] == 30);
    assert(trace[1]->trust_mode == cp::transform_execution_mode_v1::isolated_verified);
}
