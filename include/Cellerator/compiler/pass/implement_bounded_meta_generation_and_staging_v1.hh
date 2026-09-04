#pragma once

#include <Cellerator/compiler/pass/freeze_the_pass_pipeline_stage_taxonomy_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compiler::pass::v1 {

struct meta_transform_v1;
struct meta_generation_context_v1 {
    pipeline_phase_v1 phase = pipeline_phase_v1::source_canonicalization;
    std::uint32_t depth = 0;
    void* user_data = nullptr;
};
using meta_transform_run_v1 = bool (*)(const meta_generation_context_v1&,
    std::vector<meta_transform_v1>&) noexcept;

struct meta_transform_v1 {
    std::string name;
    pipeline_phase_v1 phase = pipeline_phase_v1::source_canonicalization;
    meta_transform_run_v1 run = nullptr;
    void* user_data = nullptr;
};

struct meta_generation_policy_v1 {
    std::uint32_t maximum_generated_transforms = 32;
    std::uint32_t maximum_depth = 4;
};

enum class meta_generation_status_v1 : std::uint8_t {
    success = 0,
    invalid_transform,
    generation_failed,
    phase_violation,
    generation_limit,
    depth_limit,
    cycle,
};

struct meta_generation_receipt_v1 {
    meta_generation_status_v1 status = meta_generation_status_v1::success;
    std::vector<std::string> execution_order;
    std::string diagnostic;
};

[[nodiscard]] meta_generation_receipt_v1 run_bounded_meta_generation_v1(
    const std::vector<meta_transform_v1>& roots,
    meta_generation_policy_v1 policy = {});

}  // namespace cellerator::compiler::pass::v1
