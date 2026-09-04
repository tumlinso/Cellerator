#include <Cellerator/compiler/driver/implement_plain_c_passthrough_planning_v1.hh>
#include <iostream>
int main(int argc, char** argv) { if (argc < 2) return 2; std::vector<std::string> args(argv + 2, argv + argc); const auto plan = cellerator::compiler::driver::plan_plain_cxx_passthrough_v1(argv[1], args, false); std::cout << plan.compiler; for (const auto& arg : plan.arguments) std::cout << ' ' << arg; std::cout << '\n'; }
