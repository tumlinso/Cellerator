#include <Cellerator/compiler/driver/forward_and_remap_downstream_diagnostics_v1.hh>
#include <iostream>
int main() { using namespace cellerator::compiler::driver; auto out = remap_downstream_diagnostic_v1({downstream_severity_v1::error, {"shadow.cc", 12, 4}, {"shadow.cc", 12, 8}, "error", {}, 1}, {{"shadow.cc", "source.cell", 10, 20, 3}}); std::cout << out.begin.file << ':' << out.begin.line << ':' << out.begin.column << ':' << out.message << '\n'; return out.downstream_exit_code; }
