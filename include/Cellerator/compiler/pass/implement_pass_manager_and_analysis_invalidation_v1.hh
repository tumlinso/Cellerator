#pragma once

#include <Cellerator/compiler/pass/freeze_the_pass_pipeline_stage_taxonomy_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compiler::pass::v1 {

using analysis_set_v1 = std::uint64_t;

enum class pass_status_v1 : std::uint8_t {
    success = 0,
    invalid_pipeline,
    missing_required_analysis,
    cancelled,
    pass_failed,
};

struct pass_context_v1 {
    std::uint64_t module_revision = 0;
    analysis_set_v1 available_analyses = 0;
    std::uint32_t scope_depth = 0;
    void* user_data = nullptr;
};

struct pass_result_v1 {
    bool changed = false;
    analysis_set_v1 preserved_analyses = 0;
    analysis_set_v1 produced_analyses = 0;
    std::string diagnostic;
};

using pass_run_v1 = bool (*)(pass_context_v1&, pass_result_v1&) noexcept;
using pass_cancelled_v1 = bool (*)(void*) noexcept;

struct pass_descriptor_v1 {
    std::string name;
    pipeline_stage_v1 stage{};
    std::uint32_t scope_depth = 0;
    analysis_set_v1 required_analyses = 0;
    pass_run_v1 run = nullptr;
};

struct pass_timing_v1 {
    std::string name;
    std::uint64_t nanoseconds = 0;
};

struct pass_pipeline_receipt_v1 {
    pass_status_v1 status = pass_status_v1::success;
    std::vector<std::string> ordered_passes;
    std::vector<pass_timing_v1> timings;
    std::vector<std::string> diagnostics;
    analysis_set_v1 final_analyses = 0;
    std::uint64_t deterministic_replay_hash = 0;
};

[[nodiscard]] pass_pipeline_receipt_v1 run_pass_pipeline_v1(
    const std::vector<pass_descriptor_v1>& pipeline,
    pass_context_v1 context,
    pass_cancelled_v1 cancelled = nullptr,
    void* cancellation_context = nullptr) noexcept;

}  // namespace cellerator::compiler::pass::v1
