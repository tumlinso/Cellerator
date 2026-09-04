#include <Cellerator/compiler/tooling/deliver_celleratord_c_parity_milestone_v1.hh>

#include <cassert>

using namespace Cellerator::compiler::tooling;

int main() {
    const compile_command_input_v1 command{
        "/workspace", "mixed.cell",
        "cellerator -resource-dir /opt/cellerator/clang/18 -c mixed.cell", {}};
    const auto resolved = resolve_compile_command_v1(command);
    assert(resolved && resolved->resource_directory == "/opt/cellerator/clang/18");

    const language_feature_item_v1 cpp{"vector", "std::vector", "plain.cc", 0, 6, false};
    const language_feature_item_v1 cell{"Gene", "domain", "mixed.cell", 8, 12, true};
    const auto completion = merge_language_features_v1(
        language_feature_v1::completion, {cpp}, {cell});
    const auto navigation = merge_language_features_v1(
        language_feature_v1::definition, {cpp}, {cell});
    assert(completion.items.size() == 2 && navigation.items.size() == 2);

    const tooling_diagnostic_v1 cpp_diagnostic{
        tooling_diagnostic_phase_v1::clangd,
        tooling_diagnostic_severity_v1::error, {0, 1}, "CXX", "C++ diagnostic", {}, {}};
    const tooling_diagnostic_v1 cell_diagnostic{
        tooling_diagnostic_phase_v1::source,
        tooling_diagnostic_severity_v1::error, {2, 3}, "CELL", "field syntax", {}, {}};
    assert(merge_diagnostics_v1({cpp_diagnostic}, {cell_diagnostic}).size() == 2);

    bool launched = false;
    bool stopped = false;
    clangd_worker_lifecycle_v1 worker({
        [](const std::string&) { return std::string("/usr/bin/clangd"); },
        [](const std::string&) { return std::string("clangd version 18.1.0"); },
        [&](const std::string&, const std::vector<std::string>&) {
            launched = true;
            return true;
        },
        [&] { stopped = true; }});
    assert(worker.start("clangd", {}).status == clangd_worker_status_v1::running);
    worker.stop();
    assert(launched && stopped &&
           worker.diagnostic().status == clangd_worker_status_v1::stopped);

    const auto host_capabilities = editor_capabilities_v1(false, false);
    assert(host_capabilities[0].available && host_capabilities[1].available);
    assert(!host_capabilities.back().available);

    celleratord_cpp_parity_receipt_v1 receipt{
        "bin/celleratord", resolved->resource_directory, 1, 1,
        true, true, true, true, true, stopped};
    assert(validate_celleratord_cpp_parity_v1(receipt) ==
           celleratord_cpp_parity_status_v1::valid);
    receipt.worker_process_stopped = false;
    assert(validate_celleratord_cpp_parity_v1(receipt) ==
           celleratord_cpp_parity_status_v1::process_leaked);
}
