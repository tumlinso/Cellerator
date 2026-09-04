#pragma once

#include <cstdint>

namespace cellerator::compiler::diagnostics::v1 {

enum class validation_mode : std::uint8_t {
    verified = 0,
    checked,
    trusted,
    unsafe,
    unchecked,
};

enum class validation_domain : std::uint8_t {
    parsing = 0,
    semantic_invariants,
    exact_coverage,
    numerical_claims,
    resources,
    native_bindings,
    backend_support,
};

enum class validation_failure_policy : std::uint8_t {
    mandatory_failure = 0,
    overridable_diagnostic,
    accepted_trust_assertion,
    skipped,
};

// Parseability, exact contribution ownership, resource feasibility and backend
// support are never weakened inside a Cellerator plan. Unchecked execution is
// ordinary C++/CUDA outside this validated compiler pipeline.
[[nodiscard]] validation_failure_policy validation_policy(
    validation_mode mode,
    validation_domain domain) noexcept;

} // namespace cellerator::compiler::diagnostics::v1
