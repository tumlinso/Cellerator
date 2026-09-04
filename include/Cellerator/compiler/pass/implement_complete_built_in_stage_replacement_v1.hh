#pragma once

#include <Cellerator/compiler/pass/freeze_the_pass_pipeline_stage_taxonomy_v1.hh>

#include <cstdint>
#include <string>

namespace cellerator::compiler::pass::v1 {

enum class stage_replacement_policy_v1 : std::uint8_t {
    prefer_replacement_with_fallback = 0,
    force_replacement,
};

enum class stage_replacement_status_v1 : std::uint8_t {
    success = 0,
    invalid_stage,
    missing_implementation,
    replacement_failed,
    built_in_failed,
};

struct stage_replacement_context_v1 {
    pipeline_phase_v1 phase = pipeline_phase_v1::source_canonicalization;
    void* stage_state = nullptr;
    std::string diagnostic;
};

using stage_implementation_v1 = bool (*)(stage_replacement_context_v1&) noexcept;

struct stage_replacement_request_v1 {
    pipeline_phase_v1 phase = pipeline_phase_v1::source_canonicalization;
    stage_implementation_v1 built_in = nullptr;
    stage_implementation_v1 replacement = nullptr;
    stage_replacement_policy_v1 policy =
        stage_replacement_policy_v1::prefer_replacement_with_fallback;
    void* stage_state = nullptr;
};

struct stage_replacement_receipt_v1 {
    stage_replacement_status_v1 status = stage_replacement_status_v1::success;
    bool replacement_attempted = false;
    bool replacement_selected = false;
    bool fallback_used = false;
    std::string diagnostic;
};

[[nodiscard]] stage_replacement_receipt_v1 run_stage_replacement_v1(
    const stage_replacement_request_v1& request) noexcept;

}  // namespace cellerator::compiler::pass::v1
