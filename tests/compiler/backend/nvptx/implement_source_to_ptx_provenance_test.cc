#include <Cellerator/compiler/backend/nvptx/implement_source_to_ptx_provenance_v1.hh>

#include <cassert>
#include <iostream>

using namespace Cellerator::compiler::backend::nvptx;

int main() {
    const std::string digest =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    source_to_ptx_provenance_sidecar_v1 sidecar;
    sidecar.ptx_sha256 = digest;
    assembled_resource_provenance_v1 resource;
    resource.identity = 701u;
    resource.resources = {12u, 0u, 0u, 0u, 128u};
    resource.assembler_version = "ptxas 12.0.140";
    resource.image_sha256 = digest;
    sidecar.resource_reports = {resource};

    source_to_ptx_provenance_entry_v1 relation;
    relation.source = {101u, 102u, "biology/relation.ce", 14u, 5u, 18u, 2u, digest};
    relation.semantic_node_identity = 201u;
    relation.planning_node_identity = 301u;
    relation.realization_node_identity = 401u;
    relation.ptx_begin_line = 22u;
    relation.ptx_end_line = 29u;
    relation.resource_report_identity = 701u;
    sidecar.entries = {relation};

    std::string diagnostic;
    assert(validate_source_to_ptx_provenance_v1(sidecar, &diagnostic) ==
           source_to_ptx_provenance_status_v1::success);
    const auto forward = find_ptx_for_source_operation_v1(sidecar, 101u, 102u);
    assert(forward.size() == 1u && forward[0]->semantic_node_identity == 201u &&
           forward[0]->planning_node_identity == 301u &&
           forward[0]->realization_node_identity == 401u &&
           forward[0]->ptx_begin_line == 22u);
    const auto* reverse = find_source_for_ptx_line_v1(sidecar, 25u);
    assert(reverse != nullptr && reverse->source.path == "biology/relation.ce" &&
           reverse->source.operation_identity == 102u);
    const auto* report = find_resource_report_v1(sidecar, *reverse);
    assert(report != nullptr && report->resources.registers == 12u &&
           report->resources.shared_bytes == 128u);

    sidecar.entries[0].resource_report_identity = 999u;
    assert(validate_source_to_ptx_provenance_v1(sidecar) ==
           source_to_ptx_provenance_status_v1::invalid_resource_reference);
    std::cout << "relation source -> semantic -> planning -> realization -> PTX -> source navigated\n";
}
