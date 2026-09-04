#pragma once
#include <Cellerator/execution/lowering_resumption/resumption_v1.hh>
#include <string>
#include <vector>
namespace cellerator::compiler::driver {
struct toolchain_fingerprint_input_v1 { std::string executable_content_hash, version, target, resource_directory, runtime_identity, driver_identity, backend_plugin_revision; std::vector<std::string> critical_flags; };
execution::lowering_resumption::stable_identity_v1 fingerprint_toolchain_v1(toolchain_fingerprint_input_v1);
}  // namespace cellerator::compiler::driver
