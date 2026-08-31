#include <Cellerator/compute/training/v2/generation_readiness.hh>

#include <array>
#include <cassert>

using namespace cellerator::compute::training_v2;
using namespace cellerator::execution;
using namespace cellerator::execution::training_v2;

namespace {

axis_identity axis(std::uint32_t value) {
    return {{value, 1u}, {value + 1u, 1u}, {value + 2u, 1u},
        {value + 3u, 1u}};
}

} // namespace

int main() {
    std::array<generation_component_readiness_v2, 3> components{};
    components[0] = {10u, 7u, {6u}, 101u,
        generation_component_state_v2::ready, true, {}};
    components[1] = {20u, 3u, {6u}, 102u,
        generation_component_state_v2::ready, true, {}};
    components[2] = {30u, 0u, {5u}, 0u,
        generation_component_state_v2::unavailable, false, {}};
    const generation_publication_v2 publication{{1u, 1u}, {2u}, {5u}, {6u},
        axis(10u), axis(20u), training_order_mode_v2::persistent_physical,
        false, {}, components.size(), components.data()};
    generation_publication_receipt_v2 receipt{};
    assert(validate_generation_readiness_v2(publication, receipt));
    assert(receipt.required_component_count == 2u
        && receipt.occupied_slot_count == 10u
        && !receipt.canonicalized);
    value_generation current{5u};
    assert(publish_ready_generation_v2(publication, current, receipt));
    assert(current.value == 6u);

    current = {5u};
    components[1].state = generation_component_state_v2::preparing;
    assert(!publish_ready_generation_v2(publication, current, receipt));
    assert(current.value == 5u);
    components[1].state = generation_component_state_v2::ready;
    components[1].component_identity = 10u;
    assert(!validate_generation_readiness_v2(publication, receipt));
}
