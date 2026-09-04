#include <Cellerator/compiler/driver/deliver_the_driver_passthrough_milestone_v1.hh>
#include <iostream>
int main(int argc, char** argv) { if (argc < 3 || std::string_view(argv[1]) != "--driver") { std::cerr << "usage: cellerator --driver COMPILER [ARG...]\n"; return 2; } std::vector<std::string> args(argv + 3, argv + argc); return cellerator::compiler::driver::run_driver_passthrough_v1(argv[2], args).downstream_exit_code; }
