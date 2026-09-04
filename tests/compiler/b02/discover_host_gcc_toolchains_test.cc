#include <Cellerator/compiler/driver/discover_host_gcc_toolchains_v1.hh>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
using namespace cellerator::compiler::driver;
int main() {
    gcc_discovery_input_v1 input{{"/missing", "/usr"}, "x86_64-linux-gnu", "gcc-system", "cxx11"};
    const auto out = discover_host_gcc_v1(input, [](std::string_view p){ return std::filesystem::exists(p); });
    if (!out || out.cxx != "/usr/bin/g++" || out.cc != "/usr/bin/gcc" || out.linker.empty() || out.include_root.empty() || out.libstdcxx_abi_mode != "cxx11") return EXIT_FAILURE;
    const auto temporary = std::filesystem::temp_directory_path() / "ce_ccp1_b02_004_probe.cc";
    const auto object = std::filesystem::temp_directory_path() / "ce_ccp1_b02_004_probe.o";
    { std::ofstream source(temporary); source << "int cellerator_gcc_probe() { return 4; }\n"; }
    const auto command = out.cxx + " -std=c++17 -c " + temporary.string() + " -o " + object.string();
    const int status = std::system(command.c_str());
    const bool valid = status == 0 && std::filesystem::file_size(object) > 0;
    std::filesystem::remove(temporary); std::filesystem::remove(object);
    if (!valid) return EXIT_FAILURE;
    std::cout << "validated independently discovered GCC plain-object route\n";
}
