#include <Cellerator/compiler/backend/nvcc/map_cuda_diagnostics_and_line_information_v1.hh>

#include <cassert>
#include <string>
#include <vector>

int main() {
    using namespace cellerator::compiler::backend::nvcc::v1;

    assert(emit_line_directive(17, "model.cell") ==
           "#line 17 \"model.cell\"\n");
    assert(emit_line_directive(0, "model.cell").empty());

    const auto flags = diagnostic_nvcc_options({true, true, true, "tmp/nvcc"});
    assert((flags == std::vector<std::string>{
        "-lineinfo", "--resource-usage", "--keep", "--keep-dir=tmp/nvcc"}));

    const std::vector<cuda_source_span> map{
        {"kernel.cu", 20, 24, "model.cell", 70, 991}};
    const auto mapped = map_cuda_diagnostic(
        "kernel.cu:22:9: error: identifier is undefined", map);
    assert(mapped);
    assert(mapped->source_file == "model.cell");
    assert(mapped->source_line == 72);
    assert(mapped->column == 9);
    assert(mapped->ir_node == 991);
    assert(mapped->severity == "error");
    assert(mapped->message == "identifier is undefined");
    assert(!map_cuda_diagnostic("kernel.cu:30:1: error: outside map", map));
}
