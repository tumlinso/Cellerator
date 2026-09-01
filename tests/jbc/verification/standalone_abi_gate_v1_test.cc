#include <Cellerator/execution/geometry_acquisition_v2/schema.hh>
#include <Cellerator/profiling/joint_compiler/manifest_v1.hh>

#include <cassert>
#include <type_traits>

namespace acquisition = cellerator::execution::acquisition_v2;
namespace profile = cellerator::profiling::joint_compiler;

static_assert(acquisition::schema_version == 2u);
static_assert(std::is_standard_layout_v<acquisition::acquisition_request>);
static_assert(std::is_trivially_copyable_v<acquisition::acquisition_request>);
static_assert(std::is_standard_layout_v<profile::mechanism_manifest_v1>);
static_assert(std::is_trivially_copyable_v<profile::mechanism_manifest_v1>);
static_assert(std::is_standard_layout_v<profile::atom_profile_record_v1>);
static_assert(std::is_trivially_copyable_v<profile::atom_profile_record_v1>);

int main() {
    profile::mechanism_manifest_v1 manifest{};
    manifest.mechanism_identity = {1u, 1u};
    manifest.candidate_id = 2u;
    manifest.kernel_id = 3u;
    manifest.useful_interactions = 4u;
    manifest.launch_count = 1u;
    const profile::atom_profile_record_v1 record{
        {5u, 1u}, {6u, 1u}, 7u, 8u, &manifest, 1u};
    std::uint64_t emitted = 0u;
    const profile::profile_export_sink_v1 sink{
        &emitted,
        [](void *context, const profile::atom_profile_record_v1 &) noexcept {
            ++*static_cast<std::uint64_t *>(context);
            return true;
        }};
    assert(profile::export_atom_profile_manifest_v1(&record, 1u, sink));
    assert(emitted == 1u);
}
