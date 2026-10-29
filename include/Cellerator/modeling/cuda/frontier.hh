#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

namespace cellerator::modeling::cuda {

// Device-resident CSR reverse dependencies. Upload DependencyIncidence::offsets
// and ::ordinals from incremental.hh; changed_sources contains dense source
// ordinals obtained from DependencyIncidence::source_ordinal. Entity IDs and
// roles are never cast to dense ordinals. Retain the immutable relation snapshot
// associated with this projection until device work completes.
struct FrontierPostings {
    const std::uint64_t* offsets = nullptr;       // source_count + 1
    const std::uint64_t* interaction_ordinals = nullptr; // posting_count
    std::uint64_t source_count = 0;
    std::uint64_t posting_count = 0;
    std::uint64_t interaction_count = 0;
};

enum FrontierFlag : std::uint32_t {
    frontier_success = 0,
    frontier_invalid_source = 1,
    frontier_invalid_posting = 2,
    frontier_insufficient_capacity = 4,
};

struct FrontierStatus {
    std::uint64_t required = 0; // exact union size unless an input error occurred
    std::uint64_t written = 0;  // zero on ANY flag: never consume a partial result
    std::uint32_t flags = 0;
    std::uint32_t reserved = 0;
};

struct FrontierWorkspace {
    std::uint32_t* claims = nullptr; // cleared by each call; one per interaction
    std::uint64_t capacity = 0;
};

struct FrontierOutput {
    std::uint64_t* ordinals = nullptr;
    std::uint64_t capacity = 0;
    FrontierStatus* status = nullptr;
};

// Union changed-source postings into a compact deduplicated worklist. Order is
// unspecified; map ordinals through the retained relation, or sort explicitly
// if an execution consumer requires canonical order. No dense all-pairs scan.
// Costs: O(interaction_count) claim clearing plus selected postings and changes.
// No allocation, device switch, implicit synchronization, or host status copy.
// All pointers are caller-owned device memory on the current device, valid
// until stream completion. Buffers must be disjoint and workspace/status must
// not be reused concurrently. Sequential calls on one stream are supported.
// Pointer/capacity errors return synchronously. Device content errors set flags.
// Output storage may be touched on a flagged result, but written is zero; retry
// overflow with sufficient capacity. Eligibility is a separate later operation:
// do not discard invalidations just because an interaction is currently gated.
cudaError_t build_changed_frontier(FrontierPostings postings,
                                   const std::uint64_t* changed_sources,
                                   std::uint64_t changed_count,
                                   FrontierWorkspace workspace,
                                   FrontierOutput output,
                                   cudaStream_t stream) noexcept;

} // namespace cellerator::modeling::cuda
