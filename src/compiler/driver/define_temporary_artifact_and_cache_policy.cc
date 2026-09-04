#include <Cellerator/compiler/driver/define_temporary_artifact_and_cache_policy_v1.hh>
#include <stdexcept>
namespace cellerator::compiler::driver {
artifact_policy_v1 define_artifact_policy_v1(const artifact_policy_input_v1& in) { if (in.temporary_root.empty() || in.cache_root.empty() || in.action_identity.empty() || in.content_hash.empty() || in.action_identity.find("..") != std::string::npos || in.content_hash.find("..") != std::string::npos) throw std::invalid_argument("safe roots and stable action/content identities are required"); const bool retain = in.keep == keep_temps_v1::always || (in.keep == keep_temps_v1::on_failure && in.action_failed) || (in.keep == keep_temps_v1::diagnostics && in.diagnostic_requested); return {in.temporary_root + "/action-" + in.action_identity, in.cache_root + "/sha256/" + in.content_hash.substr(0, 2) + "/" + in.content_hash, retain, !retain}; }
}  // namespace cellerator::compiler::driver
