#pragma once
#include <Cellerator/execution/program/program_v2.h>
#include <cstdint>
#include <vector>
namespace cellerator::execution::program {
enum class workspace_status { success, invalid_argument, overflow, insufficient_capacity, allocation_failure };
// Inclusive stage lifetimes. A caller must extend last through every consumer;
// asynchronous external readers are fenced by the native session owner.
struct scratch_lifetime_v2 {
    std::uint64_t bytes=0, alignment=1, first=0, last=0;
};
struct workspace_slot_v2 { scratch_lifetime_v2 lifetime{}; std::uint64_t offset=0; };
// Preparation-only owning metadata. Payload storage remains in the native
// session or caller's existing host buffer. Execution never grows this vector.
struct workspace_plan_v2 {
    std::vector<workspace_slot_v2> slots;
    std::uint64_t bytes=0, alignment=1, stage_count=0;
};
workspace_status prepare_workspace_v2(const scratch_lifetime_v2*,std::uint64_t count,
    std::uint64_t stage_count,workspace_plan_v2&) noexcept;
// First stage_count slots are canonical stage scratch; extra slots describe
// intermediates retained across multiple stages. No operation graph is inferred.
workspace_status prepare_program_workspace_v2(const prepared_program_v2&,
    const scratch_lifetime_v2* extra,std::uint64_t count,workspace_plan_v2&) noexcept;
workspace_status workspace_slice_v2(const workspace_plan_v2&,std::uint64_t slot,
    void* storage,std::uint64_t capacity,void*& output,std::uint64_t& bytes) noexcept;
// Two disjoint fixed state buffers; advancing changes roles, never copies or
// canonicalizes state. Caller commits only after its execution completes and
// all borrowers of the next write buffer have returned.
struct state_ping_pong_v2 {
    void* buffers[2]{};
    std::uint64_t bytes=0;
    unsigned current=0;
    void* input() const noexcept { return buffers[current]; }
    void* output() const noexcept { return buffers[current^1u]; }
    void commit() noexcept { current^=1u; }
};
workspace_status bind_state_ping_pong_v2(void*,std::uint64_t,void*,std::uint64_t,
    std::uint64_t required,state_ping_pong_v2&) noexcept;
} // namespace cellerator::execution::program
