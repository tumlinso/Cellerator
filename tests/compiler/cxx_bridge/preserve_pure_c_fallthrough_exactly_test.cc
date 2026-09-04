#include <Cellerator/compiler/frontend/cxx/preserve_pure_c_fallthrough_exactly_v1.hh>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <unistd.h>

namespace cxx = Cellerator::compiler::frontend::cxx;

std::string quote(const std::string& value) {
    std::string result = "'";
    for (const char character : value) result += character == '\'' ? "'\\''" : std::string(1, character);
    return result + "'";
}

int run(const std::vector<std::string>& arguments) {
    std::string command;
    for (const auto& argument : arguments) command += (command.empty() ? "" : " ") + quote(argument);
    return std::system(command.c_str());
}

int main() {
    const auto root = std::filesystem::path("/tmp") /
        ("ce_ccp1_b04_012_" + std::to_string(static_cast<long long>(getpid())));
    std::filesystem::create_directories(root);
    const auto source_path = root / "pure.cc";
    std::ofstream(source_path) << "#include <vector>\nint main(){std::vector<int> v{42};return v[0]!=42;}\n";
    for (const std::string compiler : {"/usr/bin/g++", "/usr/bin/clang++-18"}) {
        const auto object = root / (std::filesystem::path(compiler).filename().string() + ".o");
        const auto depfile = object.string() + ".d";
        const auto executable = object.string() + ".exe";
        cxx::pure_cxx_fallthrough_request_v1 request;
        request.source = "// #pragma cellerator is inert in a comment\n";
        request.original_driver_arguments = {
            compiler, "-std=c++17", "-MMD", "-MF", depfile,
            "-c", source_path.string(), "-o", object.string()};
        cxx::pure_cxx_fallthrough_plan_v1 plan;
        if (cxx::plan_pure_cxx_fallthrough_v1(request, &plan) !=
                cxx::pure_cxx_fallthrough_status_v1::success ||
            plan.mode != cxx::pure_cxx_fallthrough_mode_v1::direct_driver ||
            plan.construct_cellerator_ast_or_ir ||
            plan.forwarded_driver_arguments != request.original_driver_arguments ||
            run(plan.forwarded_driver_arguments) != 0 ||
            run({compiler, object.string(), "-o", executable}) != 0 ||
            run({executable}) != 0 || !std::filesystem::exists(depfile)) {
            std::cerr << "pure C++ driver fallthrough diverged for " << compiler << '\n';
            return 1;
        }
    }
    cxx::pure_cxx_fallthrough_request_v1 activated;
    activated.source = "#pragma cellerator profile(pbmc3k)\n";
    activated.original_driver_arguments = {"clang++", "-c", "input.cc"};
    cxx::pure_cxx_fallthrough_plan_v1 activated_plan;
    if (cxx::plan_pure_cxx_fallthrough_v1(activated, &activated_plan) !=
            cxx::pure_cxx_fallthrough_status_v1::success ||
        !activated_plan.construct_cellerator_ast_or_ir) return 1;
    std::filesystem::remove_all(root);
    return 0;
}
