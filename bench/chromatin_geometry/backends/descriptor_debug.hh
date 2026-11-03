#pragma once

#include "../contract.hh"

namespace cgs {
// Test-only borrowed device descriptors. Inspect after prepare completion and
// outside timing. Worklist entries are prepared group ordinals; canonical raw
// identity is topology.groups[ordinal].logical_id, never an assumed identity.
struct DescriptorDebugView {
    const std::uint64_t* primal_bitmap=nullptr;
    const std::uint64_t* derivative_bitmap=nullptr;
    const std::uint64_t* primal_work=nullptr;
    const std::uint64_t* derivative_work=nullptr; // P union D
    const std::uint64_t* primal_count=nullptr;
    const std::uint64_t* derivative_count=nullptr;
    const std::uint64_t* cohort_order=nullptr;
    const std::uint32_t* primal_signature=nullptr;
    const std::uint32_t* derivative_signature=nullptr;
    std::uint64_t bitmap_words_per_cell=0;
    std::uint64_t work_capacity_per_cell=0;
    std::uint64_t cohort_entries=0;
};
bool inspect_descriptor(Backend,BodyKind,const ProblemView&,BackendOptions,
                         WorkspaceView,DescriptorDebugView*) noexcept;
} // namespace cgs
