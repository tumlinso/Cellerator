#include <Cellerator/compiler/backend/implement_generated_c_representation_v1.hh>

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace cb = cellerator::compiler::backend::v1;

int main() {
    std::string source;
    const cb::generated_cpp_module_v1 module{
        "host_pipeline", {1, 2, 3}, {}, {{"increment", "value + 1"},
                                         {"double_value", "value * 2"}}};
    assert(cb::emit_generated_cpp_v1(module, &source) ==
           cb::generated_cpp_status_v1::success);
    assert(source.find("cellerator_host_pipeline") != std::string::npos);
    assert(source.find("alignas(16)") != std::string::npos);

    const auto temporary = std::filesystem::temp_directory_path() /
        "ce_ccp1_f02_004_generated.cc";
    const auto gcc_object = temporary.string() + ".gcc.o";
    const auto clang_object = temporary.string() + ".clang.o";
    { std::ofstream file(temporary); file << source; }
    const std::string gcc = "g++ -std=c++17 -Werror -c " + temporary.string() +
        " -o " + gcc_object;
    const std::string clang = "clang++-18 -std=c++17 -Werror -c " + temporary.string() +
        " -o " + clang_object;
    assert(std::system(gcc.c_str()) == 0);
    assert(std::system(clang.c_str()) == 0);
    assert(std::filesystem::file_size(gcc_object) > 0);
    assert(std::filesystem::file_size(clang_object) > 0);
    std::filesystem::remove(temporary);
    std::filesystem::remove(gcc_object);
    std::filesystem::remove(clang_object);

    auto unsafe = module;
    unsafe.stages[0].expression = "value; system(\"bad\")";
    assert(cb::emit_generated_cpp_v1(unsafe, &source) ==
           cb::generated_cpp_status_v1::unsafe_expression);
}
