#include <Cellerator/compiler/driver/fingerprint_toolchains_for_artifacts_and_resumption_v1.hh>
#include <algorithm>
namespace cellerator::compiler::driver {
execution::lowering_resumption::stable_identity_v1 fingerprint_toolchain_v1(toolchain_fingerprint_input_v1 in) { std::sort(in.critical_flags.begin(), in.critical_flags.end()); std::uint64_t low = 1469598103934665603ull, high = 1099511628211ull; auto add = [&](const std::string& value) { for (unsigned char c : value) { low = (low ^ c) * 1099511628211ull; high = (high + c + 0x9e3779b97f4a7c15ull) ^ (high << 7) ^ (high >> 3); } low = (low ^ 0xffu) * 1099511628211ull; }; add(in.executable_content_hash); add(in.version); add(in.target); add(in.resource_directory); for (const auto& flag : in.critical_flags) add(flag); add(in.runtime_identity); add(in.driver_identity); add(in.backend_plugin_revision); return {low, high}; }
}  // namespace cellerator::compiler::driver
