#pragma once

#include <Cellerator/execution/training_program_v2/program.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::execution::training_v2 {

// Immutable capture contract. It names prepared stages but owns no stream,
// launch pointer, optimizer state, or transient workspace.
struct training_graph_capture_v2 {
    std::uint64_t program_identity = 0u;
    structure_handle structure{};
    structure_epoch epoch{};
    value_generation prepared_generation{};
    std::uint64_t stage_count = 0u;
    const std::uint64_t *stage_identities = nullptr;
    std::uint64_t graph_identity = 0u;
    bool pointer_rebind_supported = true;
    bool stream_rebind_supported = true;
    bool update_policy_owned_by_caller = true;
    bool production_promoted = false;
    std::uint8_t reserved[4]{};
};

// Mutable launch data is session/caller owned and may change between replays.
struct training_graph_launch_binding_v2 {
    value_generation generation{};
    const void *source = nullptr;
    const void *destination_gradient = nullptr;
    void *destination = nullptr;
    void *source_gradient = nullptr;
    void *value_gradient = nullptr;
    void *transient_workspace = nullptr;
    std::uint64_t transient_workspace_bytes = 0u;
    std::uint64_t stream_token = 0u;
};

// The training core sees only an opaque caller policy identity and a prepared
// generic update candidate. It does not name SGD, Adam, a loss, or a framework.
struct caller_update_policy_binding_v2 {
    std::uint64_t caller_policy_identity = 0u;
    std::uint64_t prepared_update_candidate_identity = 0u;
    const void *caller_policy_state = nullptr;
    std::uint64_t caller_policy_state_bytes = 0u;
};

struct graph_capture_receipt_v2 {
    std::uint64_t validated_stage_count = 0u;
    bool pointers_rebound = false;
    bool stream_rebound = false;
    bool reprepare_required = false;
    bool update_policy_separate = true;
    std::uint8_t reserved[4]{};
};

training_result_v2 validate_training_graph_capture_v2(
    const training_graph_capture_v2 &capture,
    const training_program_v2 &program,
    graph_capture_receipt_v2 &receipt) noexcept;

training_result_v2 validate_training_graph_rebind_v2(
    const training_graph_capture_v2 &capture,
    const training_graph_launch_binding_v2 &previous,
    const training_graph_launch_binding_v2 &next,
    const caller_update_policy_binding_v2 &update_policy,
    graph_capture_receipt_v2 &receipt) noexcept;

static_assert(std::is_trivially_copyable<training_graph_capture_v2>::value,
    "training graph contract must remain trivially copyable");
static_assert(
    std::is_trivially_copyable<training_graph_launch_binding_v2>::value,
    "training graph launch binding must remain trivially copyable");

} // namespace cellerator::execution::training_v2
