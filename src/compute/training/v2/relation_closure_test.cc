#include <Cellerator/compute/training/v2/relation_closure.hh>

#include <array>
#include <cassert>

using namespace cellerator;
using namespace cellerator::compute::training_v2;
using namespace cellerator::execution;

namespace {

axis_identity axis(std::uint32_t value) {
    return {{value, 1u}, {value + 1u, 1u}, {value + 2u, 1u},
        {value + 3u, 1u}};
}

} // namespace

int main() {
    const axis_identity source_axis = axis(11u);
    const axis_identity destination_axis = axis(21u);
    const std::array<projection_edge_v2, 3> forward{{
        {0u, 0u, 0u, 0u}, {1u, 0u, 1u, 2u}, {0u, 1u, 2u, 3u}}};
    const std::array<projection_edge_v2, 3> transpose{{
        forward[0], forward[2], forward[1]}};
    const std::array<float, 5> values{{2.0F, 91.0F, 3.0F, 4.0F, 92.0F}};
    const projection_relation_v2 relation{{31u, 1u}, {32u}, {33u}, source_axis,
        destination_axis, 2u, 2u, 3u, 5u, values.data(), forward.data(),
        transpose.data()};
    std::array<std::uint64_t, 3> logical_to_forward{};
    std::array<std::uint8_t, 5> physical_seen{};
    assert(validate_projection_relation_v2(relation,
        {logical_to_forward.data(), logical_to_forward.size(), physical_seen.data(),
            physical_seen.size()}));

    const std::array<float, 2> x{{5.0F, 7.0F}};
    std::array<float, 2> y{};
    relation_closure_receipt_v2 receipt{};
    assert(relation_forward_v2(relation,
        {source_axis, training_order_mode_v2::persistent_physical, x.size(),
            x.data()},
        {destination_axis, training_order_mode_v2::persistent_physical,
            y.size(), y.data()},
        receipt));
    assert(y[0] == 31.0F && y[1] == 20.0F);

    const std::array<float, 2> dy{{11.0F, 13.0F}};
    std::array<float, 2> dx{};
    assert(relation_transpose_v2(relation,
        {destination_axis, training_order_mode_v2::persistent_physical,
            dy.size(), dy.data()},
        {source_axis, training_order_mode_v2::persistent_physical, dx.size(),
            dx.data()},
        receipt));
    assert(dx[0] == 74.0F && dx[1] == 33.0F);

    std::array<float, 5> gradients{{-1.0F, 101.0F, -1.0F, -1.0F, 102.0F}};
    assert(logical_edge_gradient_v2(relation,
        {source_axis, training_order_mode_v2::persistent_physical, x.size(),
            x.data()},
        {destination_axis, training_order_mode_v2::persistent_physical,
            dy.size(), dy.data()},
        {{31u, 1u}, {32u}, {33u}, gradients.size(), gradients.data()}, receipt));
    assert(gradients[0] == 55.0F && gradients[2] == 77.0F
        && gradients[3] == 65.0F);
    assert(gradients[1] == 101.0F && gradients[4] == 102.0F);
    assert(receipt.permanent_holes_untouched
        && receipt.physical_slots_written == 3u);

    auto duplicate = forward;
    duplicate[2].logical_edge_index = 1u;
    const projection_relation_v2 invalid{{31u, 1u}, {32u}, {33u}, source_axis,
        destination_axis, 2u, 2u, 3u, 5u, values.data(), duplicate.data(),
        transpose.data()};
    assert(!validate_projection_relation_v2(invalid,
        {logical_to_forward.data(), logical_to_forward.size(), physical_seen.data(),
            physical_seen.size()}));
}
