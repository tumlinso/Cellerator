#pragma once

#include <Cellerator/compiler/ir/realization/implement_lowering_resumption_checkpoints_v1.hh>

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace cellerator::compiler::pass::v1 {

enum class pipeline_phase_v1 : std::uint8_t {
    source_canonicalization = 0,
    profile_propagation,
    discovery,
    certification,
    decomposition,
    candidate_enumeration,
    cost_modeling,
    selection,
    realization,
    packing,
    stage_construction,
    backend_emission,
    count,
};

enum class interception_side_v1 : std::uint8_t { before = 1, after = 2 };

struct pipeline_stage_v1 {
    pipeline_phase_v1 phase = pipeline_phase_v1::source_canonicalization;
    interception_side_v1 side = interception_side_v1::before;
};

inline constexpr std::size_t pipeline_phase_count_v1 =
    static_cast<std::size_t>(pipeline_phase_v1::count);
inline constexpr std::size_t pipeline_stage_count_v1 = pipeline_phase_count_v1 * 2;

[[nodiscard]] constexpr std::uint16_t stable_stage_id_v1(
    pipeline_stage_v1 stage) noexcept {
    return static_cast<std::uint16_t>(stage.phase) * 2
        + (stage.side == interception_side_v1::after ? 1 : 0);
}

[[nodiscard]] std::string_view pipeline_phase_name_v1(
    pipeline_phase_v1 phase) noexcept;

[[nodiscard]] ir::realization::v1::ceir_facet_v1
lowering_resumption_facet_v1(pipeline_phase_v1 phase) noexcept;

[[nodiscard]] bool valid_pipeline_stage_v1(pipeline_stage_v1 stage) noexcept;

}  // namespace cellerator::compiler::pass::v1
