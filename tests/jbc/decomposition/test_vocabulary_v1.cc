#include <Cellerator/compute/decomposition/vocabulary_v1.hh>

#include <cassert>
#include <cstdint>

namespace decomposition = cellerator::compute::decomposition;

int main() {
    using decomposition::decomposition_kind_v1;
    using decomposition::fragment_role_v1;
    using decomposition::split_axis_kind_v1;

    static_assert(
        decomposition::decomposition_vocabulary_schema_version_v1 == 1u);
    static_assert(decomposition::valid_split_axis_kind_v1(
        split_axis_kind_v1::logical_edge));
    static_assert(decomposition::valid_decomposition_kind_v1(
        decomposition_kind_v1::overlapping));
    static_assert(decomposition::valid_fragment_role_v1(
        fragment_role_v1::halo));
    static_assert(!decomposition::decomposition_requires_split_axis_v1(
        decomposition_kind_v1::unsplit));
    static_assert(decomposition::decomposition_requires_split_axis_v1(
        decomposition_kind_v1::disjoint));
    static_assert(decomposition::fragment_role_owns_logical_work_v1(
        fragment_role_v1::owned));
    static_assert(!decomposition::fragment_role_owns_logical_work_v1(
        fragment_role_v1::replica));

    assert(!decomposition::valid_split_axis_kind_v1(
        static_cast<split_axis_kind_v1>(8u)));
    assert(!decomposition::valid_decomposition_kind_v1(
        static_cast<decomposition_kind_v1>(0u)));
    assert(!decomposition::valid_fragment_role_v1(
        static_cast<fragment_role_v1>(0u)));
    return 0;
}
