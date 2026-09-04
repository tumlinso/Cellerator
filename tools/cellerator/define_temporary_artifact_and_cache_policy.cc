#include <Cellerator/compiler/driver/define_temporary_artifact_and_cache_policy_v1.hh>
#include <iostream>
int main() { const auto out = cellerator::compiler::driver::define_artifact_policy_v1({"/tmp/cellerator", "/cache", "compile-1", "abcdef"}); std::cout << out.action_directory << '\n' << out.cold_cache_path << '\n'; }
