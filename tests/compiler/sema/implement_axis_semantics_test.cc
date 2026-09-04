#include <Cellerator/compiler/sema/implement_axis_semantics_v1.hh>

#include <cassert>

int main() {
    using namespace cellerator::compiler::sema::v1;
    axis_type canonical{{{1, 2}, 0, false}, 128, {3, 4}, {5, 6}, {7, 8}, 64, {9, 10}};
    auto packed = canonical;
    packed.logical_order = {11, 12};
    assert(compare_axes(canonical, canonical) == axis_compatibility::exact);
    assert(compare_axes(canonical, packed) == axis_compatibility::order_mismatch);

    auto other_domain = canonical;
    other_domain.domain.identity = {2, 1};
    assert(compare_axes(canonical, other_domain) == axis_compatibility::domain_mismatch);

    explicit_axis_mapping reorder{canonical, packed, true, true};
    assert(valid_explicit_axis_mapping(reorder));
    reorder.one_to_one = false;
    assert(!valid_explicit_axis_mapping(reorder));
}
