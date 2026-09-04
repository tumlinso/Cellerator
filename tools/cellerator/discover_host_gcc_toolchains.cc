#include <Cellerator/compiler/driver/discover_host_gcc_toolchains_v1.hh>
#include <filesystem>
#include <iostream>
int main(int argc, char** argv) { using namespace cellerator::compiler::driver; gcc_discovery_input_v1 in{{argc > 1 ? argv[1] : "/usr"}, "native", "detected", "cxx11"}; auto out = discover_host_gcc_v1(in, [](std::string_view p){ return std::filesystem::exists(p); }); if (!out) { std::cerr << out.diagnostic << '\n'; return 1; } std::cout << out.cxx << '\n'; }
