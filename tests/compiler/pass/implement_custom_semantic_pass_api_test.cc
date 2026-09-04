#include <Cellerator/compiler/pass/implement_custom_semantic_pass_api_v1.hh>

#include <cassert>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>

namespace cp = cellerator::compiler::pass::v1;
namespace cs = Cellerator::compiler::ir::semantic;

namespace {
bool validate(const cp::semantic_pass_context_v1& context) noexcept {
    return context.relation_applies->size() == 1
        && !context.profiles->empty() && !context.source_mappings->empty();
}
}  // namespace

int main() {
    const auto directory = std::filesystem::temp_directory_path()
        / "ce_ccp1_g02_004";
    std::filesystem::create_directories(directory);
    const auto source = directory / "external_pass.cc";
    const auto library = directory / "external_pass.so";
    { std::ofstream out(source); out
        << "#include <Cellerator/compiler/pass/implement_custom_semantic_pass_api_v1.hh>\n"
           "extern \"C\" bool external_replace(cellerator::compiler::pass::v1::semantic_pass_context_v1& c) noexcept {"
           "c.relation_applies->at(0).deterministic=false;"
           "c.diagnostics->push_back(\"external replacement\");"
           "(*c.analysis_cache)[\"replacement\"]=1; return true;}\n"; }
    const std::string compile = "g++ -std=c++17 -shared -fPIC -I"
        CELLERATOR_TEST_INCLUDE_ROOT " " + source.string() + " -o "
        + library.string();
    assert(std::system(compile.c_str()) == 0);
    void* handle = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
    assert(handle != nullptr);
    auto run = reinterpret_cast<cp::semantic_pass_run_v1>(
        dlsym(handle, "external_replace"));
    assert(run != nullptr);

    std::vector<cs::relation_apply_operation_ir_v1> operations(1);
    std::vector<std::string> profiles{"pbmc"};
    std::vector<cp::semantic_source_mapping_v1> mappings{{1, "model.cell", 4, 2}};
    std::vector<std::string> diagnostics;
    std::unordered_map<std::string, std::uint64_t> analyses;
    cp::semantic_pass_context_v1 context{&operations, &profiles, &mappings,
        &diagnostics, &analyses, cellerator::compiler::ir::trust_mode::checked};
    assert(cp::run_custom_semantic_pass_v1(context, run, validate)
        == cp::semantic_pass_status_v1::success);
    assert(!operations[0].deterministic && analyses["replacement"] == 1);
    assert(diagnostics[0] == "external replacement");
    dlclose(handle);
    std::filesystem::remove_all(directory);
}
