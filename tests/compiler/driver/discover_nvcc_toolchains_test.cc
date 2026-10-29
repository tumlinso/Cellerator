#include <Cellerator/compiler/driver/discover_nvcc_toolchains_v1.hh>
#include <cstdlib>
#include <iostream>
#include <set>
using namespace cellerator::compiler::driver;
int main() {
    nvcc_discovery_input_v1 in{{}, {{"/cuda11", "11.8", 8, 11, {70, 80}}, {"/cuda12", "12.6", 9, 13, {70, 90}}}, 12, 70};
    const std::set<std::string> installed{"/cuda11/bin/nvcc", "/cuda12/bin/nvcc"};
    auto out = discover_nvcc_v1(in, [&](std::string_view p){ return installed.count(std::string(p)); });
    if (out || out.diagnostic.find("host compiler major 12") == std::string::npos || out.diagnostic.find("11.8") == std::string::npos) return EXIT_FAILURE;
    in.installations.erase(in.installations.begin());
    out = discover_nvcc_v1(in, [&](std::string_view p){ return installed.count(std::string(p)); });
    if (!out || out.toolkit_root != "/cuda12" || out.ptxas.empty() || out.nvlink.empty() || out.fatbinary.empty() || out.version_identity != "12.6") return EXIT_FAILURE;
    in.requested_architecture = 75;
    out = discover_nvcc_v1(in, [&](std::string_view p){ return installed.count(std::string(p)); });
    if (out || out.diagnostic.find("unsupported") == std::string::npos) return EXIT_FAILURE;
    std::cout << "validated multiple CUDA installations and precise host/architecture rejection\n";
}
