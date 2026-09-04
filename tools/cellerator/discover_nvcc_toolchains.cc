#include <Cellerator/compiler/driver/discover_nvcc_toolchains_v1.hh>
#include <filesystem>
#include <iostream>
int main(int argc, char** argv) { using namespace cellerator::compiler::driver; nvcc_discovery_input_v1 in; if (argc > 1) in.explicit_root = argv[1]; auto out = discover_nvcc_v1(in, [](std::string_view p){ return std::filesystem::exists(p); }); if (!out) { std::cerr << out.diagnostic << '\n'; return 1; } std::cout << out.nvcc << '\n'; }
