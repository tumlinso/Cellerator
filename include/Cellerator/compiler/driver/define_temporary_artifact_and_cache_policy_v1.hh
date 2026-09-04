#pragma once
#include <string>
namespace cellerator::compiler::driver {
enum class keep_temps_v1 { never, on_failure, diagnostics, always };
struct artifact_policy_input_v1 { std::string temporary_root, cache_root, action_identity, content_hash; keep_temps_v1 keep = keep_temps_v1::never; bool action_failed = false, diagnostic_requested = false; };
struct artifact_policy_v1 { std::string action_directory, cold_cache_path; bool retain_temporary = false, cleanup_safe = false; };
artifact_policy_v1 define_artifact_policy_v1(const artifact_policy_input_v1&);
}  // namespace cellerator::compiler::driver
