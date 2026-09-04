#include <Cellerator/compiler/driver/define_compilation_database_and_dependency_file_behavior_v1.hh>
#include <iostream>
int main() { std::cout << cellerator::compiler::driver::compilation_database_entry_v1({".", "input.cc", "input.o", "input.d", {}, {"cellerator", "-c", "input.cc"}}) << '\n'; }
