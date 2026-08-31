#include <Cellerator/compute/training/v2/value_modes.hh>

#include <array>
#include <cassert>

using namespace cellerator::compute::training_v2;
using namespace cellerator::execution;
using namespace cellerator::execution::training_v2;

int main() {
    const std::array<std::uint64_t, 3> map{{0u, 2u, 4u}};
    const std::array<float, 3> logical{{2.0F, 3.0F, 5.0F}};
    std::array<float, 6> physical{{91.0F, 101.0F, 92.0F, 102.0F, 93.0F, 103.0F}};
    training_value_binding_v2 binding{{7u, 1u}, {8u}, {9u},
        training_value_mode_v2::logical_primary, logical.size(), physical.size(),
        map.data(), logical.data(), physical.data()};
    std::array<std::uint8_t, 6> seen{};
    assert(validate_training_value_binding_v2(binding,
        {seen.data(), seen.size()}));
    value_mode_receipt_v2 receipt{};
    assert(prepare_training_values_v2(binding, receipt));
    assert(physical[0] == 2.0F && physical[2] == 3.0F
        && physical[4] == 5.0F);
    assert(physical[1] == 101.0F && physical[3] == 102.0F
        && physical[5] == 103.0F);

    binding.mode = training_value_mode_v2::projection_primary;
    binding.logical_values = nullptr;
    const auto unchanged = physical;
    assert(prepare_training_values_v2(binding, receipt));
    assert(receipt.direct_projection_binding && physical == unchanged);

    std::array<float, 3> exported{};
    assert(export_logical_values_v2(
        binding, exported.data(), exported.size(), receipt));
    assert(exported == logical);
    std::array<float, 3> gradient{};
    assert(export_logical_gradients_v2(binding, physical.data(), physical.size(),
        gradient.data(), gradient.size(), receipt));
    assert(gradient == logical);

    const std::array<std::uint64_t, 3> duplicate{{0u, 2u, 2u}};
    binding.logical_to_physical = duplicate.data();
    assert(!validate_training_value_binding_v2(binding,
        {seen.data(), seen.size()}));
}
