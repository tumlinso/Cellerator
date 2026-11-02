#pragma once

#include <Cellerator/containers/posting_index.hh>
#include <Cellerator/ops/condition/program.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::execution::dependency {

// Raw per-batch CSR candidates. Each row is strictly ordered by stable edge ID
// and unique. These POD views are C++17/CUDA compatible and do not own storage.
struct CandidateBatchView {
    const std::uint64_t* offsets = nullptr;
    const containers::IndexedWorkItem* items = nullptr;
    std::uint64_t item_count = 0u;
};

struct WorksetView {
    std::uint64_t* offsets = nullptr;
    std::uint64_t offsets_capacity = 0u;
    containers::IndexedWorkItem* items = nullptr;
    std::uint64_t item_capacity = 0u;
    std::uint64_t written = 0u;
    std::uint64_t required = 0u;
    std::uint64_t structure_epoch = 0u;
    std::uint64_t state_generation = 0u;
};

struct ConditionWorksets {
    WorksetView primal;
    WorksetView derivative;
};

// Filter primal work and derivative proposals with the condition bitmap.
// Derivative work is the unique union of eligible primal candidates and
// eligible derivative proposals. Capacity validation precedes all output
// writes.
ops::condition::StatusV1 build_condition_worksets(
    const ops::condition::ProgramViewV1& program,
    std::uint64_t batch_count,
    std::uint64_t state_generation,
    const std::uint64_t* bitmap_words,
    std::uint64_t bitmap_word_count,
    const CandidateBatchView& primal_candidates,
    const CandidateBatchView& derivative_candidates,
    ConditionWorksets* output) noexcept;

static_assert(std::is_trivially_copyable<CandidateBatchView>::value,
              "candidate batches are borrowed POD views");
static_assert(std::is_trivially_copyable<WorksetView>::value,
              "worksets are borrowed POD views");
static_assert(std::is_trivially_copyable<ConditionWorksets>::value,
              "condition worksets are borrowed POD views");

} // namespace cellerator::execution::dependency
