#pragma once

#include <Cellerator/compiler/diagnostics/freeze_validation_mode_semantics_v1.hh>

#include <cstdint>

namespace cellerator::compiler::diagnostics::v1 {

enum class structural_impossibility : std::uint32_t {
    none = 0,
    malformed_graph = 1U << 0U,
    missing_required_operand = 1U << 1U,
    impossible_reference = 1U << 2U,
    uninterpretable_text = 1U << 3U,
    unrepresentable_backend_state = 1U << 4U,
};

struct structural_check_request {
    std::uint32_t failures = 0;
    validation_mode mode = validation_mode::checked;
};

struct structural_check_result {
    bool can_continue = false;
    structural_impossibility first_failure = structural_impossibility::none;
};

[[nodiscard]] structural_check_result check_structural_possibility(
    const structural_check_request& request) noexcept;

} // namespace cellerator::compiler::diagnostics::v1
