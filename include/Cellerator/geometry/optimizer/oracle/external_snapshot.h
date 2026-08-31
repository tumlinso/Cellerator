#pragma once

#include "Cellerator/geometry/optimizer/oracle/exact_oracle.h"

#include <cstddef>
#include <cstdint>

namespace cellerator::geometry::optimizer::oracle {

enum class oracle_snapshot_status : std::uint32_t {
    success = 0,
    invalid_argument,
    invalid_problem,
    capacity_exceeded,
    arithmetic_overflow,
    unsupported_encoding,
    corrupt_snapshot,
};

struct oracle_snapshot_requirements {
    oracle_snapshot_status status = oracle_snapshot_status::invalid_argument;
    std::uint64_t bytes = 0;
    std::uint64_t alignment = 8;
};

struct oracle_snapshot_view {
    const std::byte* bytes = nullptr;
    std::uint64_t byte_count = 0;
};

struct mutable_oracle_snapshot_view {
    std::byte* bytes = nullptr;
    std::uint64_t byte_capacity = 0;
};

struct parsed_oracle_snapshot {
    oracle_snapshot_status status = oracle_snapshot_status::invalid_argument;
    exact_oracle_problem_view problem{};
    std::uint64_t encoded_bytes = 0;
    std::uint64_t payload_fingerprint = 0;
};

oracle_snapshot_requirements query_oracle_snapshot_requirements(
        const exact_oracle_problem_view& problem) noexcept;

oracle_snapshot_status write_oracle_snapshot(
        const exact_oracle_problem_view& problem,
        const mutable_oracle_snapshot_view& destination,
        std::uint64_t* bytes_written) noexcept;

parsed_oracle_snapshot parse_oracle_snapshot(
        const oracle_snapshot_view& snapshot) noexcept;

struct dummy_alternative_result {
    exact_oracle_status status = exact_oracle_status::invalid_argument;
    std::int64_t objective = 0;
    std::uint32_t selected_candidate_count = 0;
    bool admissible = false;
};

// Deliberately non-optimizing external-solver fixture. It selects the complete
// pure residual fallback, proving that a separately implemented solver can
// consume the serialized problem and emit a legal candidate byte vector.
dummy_alternative_result solve_dummy_alternative(
        const exact_oracle_problem_view& problem,
        std::uint8_t* selection,
        std::uint32_t selection_capacity,
        std::uint32_t* contribution_owners,
        std::uint64_t contribution_owner_capacity) noexcept;

}  // namespace cellerator::geometry::optimizer::oracle
