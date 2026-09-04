#include <Cellerator/compiler/driver/define_toolchain_override_precedence_v1.hh>
#include <iostream>
int main(int argc, char** argv) { using namespace cellerator::compiler::driver; override_candidates_v1 in; for (int i = 1; i < argc && i <= 6; ++i) in.values[i - 1] = argv[i]; const auto out = resolve_toolchain_override_v1(in); if (!out) return 1; std::cout << override_source_name_v1(out.source) << ':' << out.value << '\n'; }
