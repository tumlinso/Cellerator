#include <Cellerator/compute/projection_family/value_gradient_identity_v1.hh>

#include <cassert>
#include <cstdint>

namespace family = cellerator::compute::projection_family;
namespace execution = cellerator::execution;

namespace {

execution::persistent_axis_identity axis(std::uint64_t base) {
    return {{execution::biological_abi_version,
             execution::serialized_record_kind::persistent_axis_identity,
             sizeof(execution::persistent_axis_identity)},
            {base + 1, base + 2}, {base + 3, base + 4},
            {base + 5, base + 6}, {base + 7, base + 8}};
}

family::support_family_identity_v1 support() {
    family::support_family_identity_v1 value{};
    value.family_identity = {1, 2};
    value.exact_support_identity = {3, 4};
    value.structure_identity = {5, 6};
    value.structure_epoch = {7};
    value.source_axis = axis(10);
    value.destination_axis = axis(30);
    value.logical_edge_order = {50, 51};
    value.logical_edge_count = (std::uint64_t{1} << 32u) + 11u;
    return value;
}

family::value_gradient_identity_v1 identity() {
    family::value_gradient_identity_v1 value{};
    value.family = support();
    value.value_identity = {60, 61};
    value.value_generation = {9};
    value.gradient_identity = {62, 63};
    value.gradient_generation = {4};
    value.logical_edge_order = value.family.logical_edge_order;
    value.logical_edge_count = value.family.logical_edge_count;
    value.flags = family::value_identity_trainable_v1
        | family::gradient_identity_present_v1;
    return value;
}

} // namespace

int main() {
    const auto current = identity();
    assert(family::validate_value_gradient_identity_v1(current).valid());

    auto next_values = current;
    next_values.value_generation.value = 10;
    assert(family::same_value_lineage_v1(current, next_values));
    assert(!family::same_value_generation_v1(current, next_values));

    auto next_gradient = current;
    next_gradient.gradient_generation.value = 5;
    assert(family::same_value_generation_v1(current, next_gradient));
    assert(!family::same_gradient_generation_v1(current, next_gradient));

    auto stale_structure = current;
    stale_structure.family.structure_epoch.value = 8;
    assert(!family::same_value_lineage_v1(current, stale_structure));

    auto wrong_order = current;
    wrong_order.logical_edge_order = {70, 71};
    assert(family::validate_value_gradient_identity_v1(wrong_order).code
           == family::value_gradient_identity_code_v1::
                  logical_edge_order_mismatch);

    auto missing_gradient = current;
    missing_gradient.gradient_identity = {};
    assert(family::validate_value_gradient_identity_v1(missing_gradient).code
           == family::value_gradient_identity_code_v1::
                  missing_gradient_identity);

    auto inference = current;
    inference.flags = 0;
    inference.gradient_identity = {};
    inference.gradient_generation = {};
    assert(family::validate_value_gradient_identity_v1(inference).valid());
}
