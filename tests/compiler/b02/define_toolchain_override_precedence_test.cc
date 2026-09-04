#include <Cellerator/compiler/driver/define_toolchain_override_precedence_v1.hh>
#include <cstdlib>
#include <iostream>
using namespace cellerator::compiler::driver;
int main() { override_candidates_v1 input; for (std::size_t winner = 0; winner != input.values.size(); ++winner) { for (std::size_t i = 0; i != input.values.size(); ++i) input.values[i] = i < winner ? "" : "value-" + std::to_string(i); const auto out = resolve_toolchain_override_v1(input); if (!out || static_cast<std::size_t>(out.source) != winner || out.value != "value-" + std::to_string(winner)) return EXIT_FAILURE; } input = {}; if (resolve_toolchain_override_v1(input) || override_source_name_v1(override_source_v1::unresolved) != "unresolved") return EXIT_FAILURE; std::cout << "validated conflicting six-level toolchain precedence matrix\n"; }
