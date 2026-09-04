#include <Cellerator/compiler/pass/implement_custom_planning_pass_api_v1.hh>

#include <cassert>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>

namespace cp = cellerator::compiler::pass::v1;

int main() {
    const auto directory = std::filesystem::temp_directory_path() / "ce_ccp1_g02_005";
    std::filesystem::create_directories(directory);
    const auto source = directory / "planning_pass.cc";
    const auto library = directory / "planning_pass.so";
    { std::ofstream out(source); out
        << "#include <Cellerator/compiler/pass/implement_custom_planning_pass_api_v1.hh>\n"
           "extern \"C\" bool plan(cellerator::compiler::pass::v1::planning_pass_context_v1& c) noexcept {"
           "c.decompositions->push_back({7,{1,2}});"
           "c.candidates->push_back({9,7,3.5,\"external\"});"
           "*c.selected_candidate=9; return true;}\n"; }
    const std::string compile = "g++ -std=c++17 -shared -fPIC -I"
        CELLERATOR_TEST_INCLUDE_ROOT " " + source.string() + " -o " + library.string();
    assert(std::system(compile.c_str()) == 0);
    void* handle = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
    assert(handle != nullptr);
    auto run = reinterpret_cast<cp::planning_pass_run_v1>(dlsym(handle, "plan"));
    std::vector<cp::planning_atom_v1> atoms{{1}, {2}};
    std::vector<cp::planning_evidence_v1> evidence{{1, 1.0}};
    std::vector<cp::planning_decomposition_v1> decompositions;
    std::vector<cp::planning_candidate_v1> candidates;
    std::uint64_t selected = 0;
    std::vector<std::string> diagnostics;
    cp::planning_pass_context_v1 context{&atoms, &evidence, &decompositions,
        &candidates, &selected, &diagnostics, cp::planning_pass_mode_v1::replace};
    assert(cp::run_custom_planning_pass_v1(context, run)
        == cp::planning_pass_status_v1::success);
    assert(selected == 9 && candidates[0].provider == "external");
    dlclose(handle);
    std::filesystem::remove_all(directory);
}
