#include <Cellerator/compiler/driver/fingerprint_toolchains_for_artifacts_and_resumption_v1.hh>
#include <iostream>
int main() { const auto id = cellerator::compiler::driver::fingerprint_toolchain_v1({"sha256", "clang-20", "x86_64", "/resource", "cuda", "driver", "plugin-1", {"-O2"}}); std::cout << std::hex << id.high << id.low << '\n'; }
