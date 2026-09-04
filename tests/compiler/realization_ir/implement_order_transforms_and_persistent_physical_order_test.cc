#include <Cellerator/compiler/ir/realization/implement_order_transforms_and_persistent_physical_order_v1.hh>
#include <cassert>
using namespace cellerator::compiler::ir::realization::v1;
int main() {
    order_identity_v1 canonical{{1, 1}, order_class_v1::canonical};
    order_identity_v1 packed{{1, 2}, order_class_v1::persistent_physical};
    order_transform_v1 gather{{2, 1}, canonical, packed,
        order_stage_kind_v1::gather, {2, 0, 1}};
    order_transform_v1 scatter{{2, 2}, packed, canonical,
        order_stage_kind_v1::scatter, {1, 2, 0}};
    persistent_order_chain_v1 chain;
    chain.relations = {{{3, 1}, packed, packed}, {{3, 2}, packed, packed}};
    chain.boundary_transforms = {gather, scatter};
    chain.transforms_reused = 1;
    assert(validate_persistent_order_chain_v1(chain) == order_status_v1::valid);
    std::vector<double> input{1, 2, 3};
    auto persistent = apply_order_transform_v1(
        apply_order_transform_v1(input, gather), scatter);
    auto materialized = apply_order_transform_v1(
        apply_order_transform_v1(apply_order_transform_v1(
            apply_order_transform_v1(input, gather), scatter), gather), scatter);
    assert(persistent == input && materialized == input);
    chain.relations[1].input = canonical;
    assert(validate_persistent_order_chain_v1(chain) ==
        order_status_v1::disconnected_chain);
}
