#include <Cellerator/profiling/joint_compiler/manifest_v1.hh>

#include <cassert>

namespace profile = cellerator::profiling::joint_compiler;

bool collect(void *raw, const profile::atom_profile_record_v1 &record) noexcept {
    auto &count = *static_cast<std::uint64_t *>(raw);
    ++count;
    return record.mechanisms[0].bytes_moved == 4096u;
}

int main() {
    const profile::mechanism_manifest_v1 mechanism{
        {1u, 1u}, 2u, 3u, 1024u, 4096u, 1u, 200u, 500u};
    const profile::atom_profile_record_v1 records[] = {
        {{4u, 1u}, {5u, 1u}, 6u, 7u, &mechanism, 1u},
        {{8u, 1u}, {5u, 1u}, 6u, 8u, &mechanism, 1u},
    };
    std::uint64_t emitted = 0u;
    const profile::profile_export_sink_v1 sink{&emitted, collect};
    assert(profile::export_atom_profile_manifest_v1(records, 2u, sink));
    assert(emitted == 2u);

    const profile::atom_profile_record_v1 duplicates[] = {
        records[0], records[0]};
    assert(profile::export_atom_profile_manifest_v1(
               duplicates, 2u, sink).code ==
        profile::export_code_v1::duplicate_atom);

    auto invalid = mechanism;
    invalid.launch_count = 0u;
    auto invalid_record = records[0];
    invalid_record.mechanisms = &invalid;
    assert(profile::export_atom_profile_manifest_v1(
               &invalid_record, 1u, sink).code ==
        profile::export_code_v1::invalid_measurement);
}
