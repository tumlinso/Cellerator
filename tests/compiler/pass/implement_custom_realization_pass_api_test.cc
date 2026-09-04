#include <Cellerator/compiler/pass/implement_custom_realization_pass_api_v1.hh>

#include <cassert>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>

namespace cp = cellerator::compiler::pass::v1;

int main() {
    const auto directory = std::filesystem::temp_directory_path() / "ce_ccp1_g02_006";
    std::filesystem::create_directories(directory);
    const auto source = directory / "realization_pass.cc";
    const auto library = directory / "realization_pass.so";
    { std::ofstream out(source); out
        << "#include <Cellerator/compiler/pass/implement_custom_realization_pass_api_v1.hh>\n"
           "extern \"C\" bool rewrite(cellerator::compiler::pass::v1::realization_pass_context_v1& c) noexcept {"
           "c.stages->push_back({2,\"external-stage\",{1}});"
           "c.bindings->push_back({2,10});"
           "c.native_fragments->push_back({2,\"external\",{1,2,3}}); return true;}\n"; }
    const std::string compile = "g++ -std=c++17 -shared -fPIC -I"
        CELLERATOR_TEST_INCLUDE_ROOT " " + source.string() + " -o " + library.string();
    assert(std::system(compile.c_str()) == 0);
    void* handle = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
    assert(handle != nullptr);
    auto run = reinterpret_cast<cp::realization_pass_run_v1>(dlsym(handle, "rewrite"));
    std::vector<cp::realization_object_v1> covers{{10, "cover"}}, projections,
        packs, targets;
    std::vector<cp::realization_stage_v1> stages{{1, "builtin", {}}};
    std::vector<cp::realization_binding_v1> bindings;
    std::vector<cp::realization_native_fragment_v1> fragments;
    std::vector<std::string> diagnostics;
    cp::realization_pass_context_v1 context{&covers, &projections, &packs,
        &stages, &bindings, &targets, &fragments, &diagnostics};
    assert(cp::run_custom_realization_pass_v1(context, run)
        == cp::realization_pass_status_v1::success);
    assert(stages.size() == 2 && fragments[0].provider == "external");
    dlclose(handle);
    std::filesystem::remove_all(directory);
}
