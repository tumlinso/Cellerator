#pragma once

#include <Cellerator/geometry/dependency/patterns.hh>

#include <cstdint>
#include <span>

namespace cellerator::geometry::dependency {

// One logical coordinate selected for a participant role. The source and
// logical order preserve the association needed to interpret coordinate;
// physical offsets and device placement belong to execution bindings.
struct CoordinateParticipant {
    role_id role{};
    SourceRef source{};
    structure::order_id order{};
    std::uint64_t coordinate = 0;
    std::uint64_t incidence_slot = 0;
};

// Decode one row-major Cartesian tuple into caller-owned storage. The final
// role varies fastest. Pattern and selection-coordinate spans are borrowed and
// must remain valid and unchanged for the duration of the call. On failure,
// output is unchanged.
Status cartesian_tuple(const CartesianPattern& pattern,
                      std::uint64_t ordinal,
                      std::span<CoordinateParticipant> output) noexcept;

// Decode a bounded row-major batch directly from the implicit product into
// caller-owned storage laid out as [tuples, roles]. `requested` must fit in
// the suffix beginning at `begin`. On failure, both output and written_tuples
// are unchanged. Pattern and selection-coordinate spans are borrowed and
// must remain valid and unchanged for the duration of the call.
Status cartesian_batch(const CartesianPattern& pattern,
                       std::uint64_t begin,
                       std::uint64_t requested,
                       std::span<CoordinateParticipant> output,
                       std::uint64_t& written_tuples) noexcept;

} // namespace cellerator::geometry::dependency
