#include <Cellerator/execution/atom_fragment/local_index_builder_v1.hh>

#include <cassert>

namespace fragment = cellerator::execution::atom_fragment;
namespace execution = cellerator::execution;

int main() {
    const std::uint64_t first[] = {1u, 4u, 7u};
    const std::uint64_t second[] = {0u, 9u};
    const std::uint64_t identities[] = {100u, 101u};
    const fragment::atom_local_component_source_v1 sources[] = {
        {1u, 10u, 11u, first, nullptr, 3u},
        {2u, 10u, 12u, second, identities, 2u},
    };
    execution::hierarchical_index_component_v1 components[2]{};
    std::uint64_t maps[5]{};
    std::uint64_t sidecars[5]{};
    const fragment::atom_local_index_buffers_v1 buffers{
        components, 2u, maps, 5u, sidecars, 5u};
    execution::hierarchical_index_space_view_v1 result{};
    assert(fragment::build_atom_local_index_space_v1(
        20u, 8u, sources, 2u, buffers, &result));
    assert(result.component_count == 2u);
    assert(result.components[0].aggregate_begin == 0u);
    assert(result.components[1].aggregate_begin == 3u);
    assert(result.components[0].index_space.local_width
        == execution::local_index_width_v1::u16);
    assert(result.components[1].index_space.global_identity_sidecar[1] == 101u);

    auto malformed = sources[1];
    const std::uint64_t duplicate[] = {3u, 3u};
    malformed.global_indices = duplicate;
    const fragment::atom_local_component_source_v1 bad_sources[] = {
        sources[0], malformed};
    assert(fragment::build_atom_local_index_space_v1(
               20u, 8u, bad_sources, 2u, buffers, &result)
               .code
        == fragment::atom_local_index_build_code_v1::
            duplicate_or_unordered_global_index);
    auto small = buffers;
    small.global_index_capacity = 4u;
    assert(fragment::build_atom_local_index_space_v1(
               20u, 8u, sources, 2u, small, &result)
               .code
        == fragment::atom_local_index_build_code_v1::insufficient_capacity);
    return 0;
}
