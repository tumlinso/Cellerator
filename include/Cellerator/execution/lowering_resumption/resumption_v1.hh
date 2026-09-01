#pragma once

#include <Cellerator/execution/geometry_acquisition_v2/schema.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::execution::lowering_resumption {

using stable_identity_v1 = acquisition_v2::stable_identity;

enum class lowering_stage_v1 : std::uint8_t {
    canonical_source = 1u,
    atom_evidence = 2u,
    semantic_atom = 3u,
    target_cover = 4u,
    physical_projection = 5u,
    packed_operand = 6u,
    executable_recipe = 7u,
    local_realization = 8u,
};

enum class compatibility_code_v1 : std::uint8_t {
    compatible = 0u,
    invalid_argument,
    corrupt_artifact,
    identity_mismatch,
    structure_epoch_mismatch,
    order_mismatch,
    target_mismatch,
    value_generation_stale,
    toolchain_mismatch,
    insufficient_capacity,
};

struct compatibility_status_v1 {
    compatibility_code_v1 code = compatibility_code_v1::compatible;
    lowering_stage_v1 inspected_stage = lowering_stage_v1::canonical_source;
    lowering_stage_v1 earliest_compatible_stage =
        lowering_stage_v1::canonical_source;
    std::uint8_t reserved[5]{};
    std::uint64_t detail = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == compatibility_code_v1::compatible;
    }
};

constexpr lowering_stage_v1 earliest_stage_for_v1(
    compatibility_code_v1 code, lowering_stage_v1 inspected) noexcept {
    switch (code) {
        case compatibility_code_v1::compatible:
            return inspected;
        case compatibility_code_v1::value_generation_stale:
            return lowering_stage_v1::packed_operand;
        case compatibility_code_v1::target_mismatch:
        case compatibility_code_v1::toolchain_mismatch:
            return lowering_stage_v1::physical_projection;
        case compatibility_code_v1::order_mismatch:
            return lowering_stage_v1::semantic_atom;
        case compatibility_code_v1::structure_epoch_mismatch:
            return lowering_stage_v1::atom_evidence;
        case compatibility_code_v1::corrupt_artifact:
        case compatibility_code_v1::identity_mismatch:
        case compatibility_code_v1::invalid_argument:
        case compatibility_code_v1::insufficient_capacity:
            return lowering_stage_v1::canonical_source;
    }
    return lowering_stage_v1::canonical_source;
}

constexpr compatibility_status_v1 make_status_v1(
    compatibility_code_v1 code, lowering_stage_v1 inspected,
    std::uint64_t detail = 0u) noexcept {
    return {code, inspected, earliest_stage_for_v1(code, inspected), {}, detail};
}

static_assert(std::is_trivially_copyable_v<compatibility_status_v1>);

}  // namespace cellerator::execution::lowering_resumption
