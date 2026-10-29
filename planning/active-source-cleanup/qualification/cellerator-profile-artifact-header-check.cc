#include <Cellerator/compiler/profile/profile_artifact_v1.hh>
#include <type_traits>
#include <limits>
namespace p = cellerator::compiler::profile::v1;
static_assert(std::is_same_v<decltype(p::profile_artifact_v1{}.revision), decltype(p::profile_artifact_header_v1{}.evidence_revision)>);
static_assert(std::is_standard_layout_v<p::profile_artifact_v1>);
static_assert(std::is_trivially_copyable_v<p::profile_artifact_v1>);
int main() { p::profile_artifact_v1 artifact; if (artifact.revision != 0u) return 1; artifact.revision = std::numeric_limits<std::uint64_t>::max(); return artifact.revision != std::numeric_limits<std::uint64_t>::max(); }
