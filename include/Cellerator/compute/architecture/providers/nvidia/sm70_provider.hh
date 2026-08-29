#pragma once

#include <Cellerator/compute/architecture/provider.hh>

namespace cellerator::compute::architecture::providers::nvidia {

inline constexpr architecture_identity_v1 sm70_provider_identity_v1{
    0x6e76696469615f73ull, 0x6d37305f70726f76ull};
inline constexpr architecture_identity_v1 sm70_wmma_f16_f32_identity_v1{
    0x736d37305f776d6dull, 0x615f663136663332ull};
inline constexpr architecture_identity_v1
    sm70_wmma_f16_memory_interface_identity_v1{
        0x736d37305f776d6dull, 0x615f6d656d5f7631ull};

const matrix_memory_interface_v1 &
sm70_wmma_f16_memory_interface_v1() noexcept;

const matrix_engine_capability_v1 &
sm70_wmma_f16_f32_capability_v1() noexcept;

const architecture_provider_v1 &sm70_provider_v1() noexcept;

provider_status_v1 register_sm70_provider_v1(
    architecture_provider_registry_v1 *registry) noexcept;

} // namespace cellerator::compute::architecture::providers::nvidia
