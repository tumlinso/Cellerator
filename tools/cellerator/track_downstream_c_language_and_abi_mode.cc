#include <Cellerator/compiler/driver/track_downstream_c_language_and_abi_mode_v1.hh>
#include <iostream>
int main(int argc, char** argv) { std::vector<std::string> args(argv + 1, argv + argc); const auto out = cellerator::compiler::driver::track_downstream_language_and_abi_v1(args); std::cout << "implementation=" << out.implementation_standard << " downstream=" << out.language_standard << '\n'; }
