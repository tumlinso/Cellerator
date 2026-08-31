#pragma once

#include <Cellerator/execution/training_program_v2/program.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::training_v2 {

using execution::axis_identity;
using execution::structure_epoch;
using execution::structure_handle;
using execution::training_v2::training_order_mode_v2;
using execution::training_v2::training_result_v2;
using execution::value_generation;

enum class generation_component_state_v2 : std::uint8_t {
    unavailable = 0u,
    preparing = 1u,
    ready = 2u
};

// Components are sorted by component_identity. A completion token is supplied
// by the execution session after its explicit stream/event dependency is met.
struct generation_component_readiness_v2 {
    std::uint64_t component_identity = 0u;
    std::uint64_t occupied_slot_count = 0u;
    value_generation generation{};
    std::uint64_t completion_token = 0u;
    generation_component_state_v2 state =
        generation_component_state_v2::unavailable;
    bool required = true;
    std::uint8_t reserved[6]{};
};

struct generation_publication_v2 {
    structure_handle structure{};
    structure_epoch epoch{};
    value_generation expected_current{};
    value_generation pending{};
    axis_identity source_axis{};
    axis_identity destination_axis{};
    training_order_mode_v2 persistent_order =
        training_order_mode_v2::persistent_physical;
    bool canonicalization_requested = false;
    std::uint8_t reserved[6]{};
    std::uint64_t component_count = 0u;
    const generation_component_readiness_v2 *components = nullptr;
};

struct generation_publication_receipt_v2 {
    value_generation published{};
    std::uint64_t required_component_count = 0u;
    std::uint64_t occupied_slot_count = 0u;
    training_order_mode_v2 retained_order =
        training_order_mode_v2::persistent_physical;
    bool canonicalized = false;
    std::uint8_t reserved[7]{};
};

training_result_v2 validate_generation_readiness_v2(
    const generation_publication_v2 &publication,
    generation_publication_receipt_v2 &receipt) noexcept;

// Publication mutates only caller/session-owned generation state after the
// entire required component set validates. No partial publication occurs.
training_result_v2 publish_ready_generation_v2(
    const generation_publication_v2 &publication,
    value_generation &caller_generation,
    generation_publication_receipt_v2 &receipt) noexcept;

static_assert(
    std::is_trivially_copyable<generation_publication_v2>::value,
    "generation publication must remain a trivially copyable cold view");

} // namespace cellerator::compute::training_v2
