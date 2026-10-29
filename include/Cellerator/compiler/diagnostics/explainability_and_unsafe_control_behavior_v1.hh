#pragma once
#include <Cellerator/compiler/diagnostics/diagnostics_v1.hh>
#include <Cellerator/compiler/diagnostics/provenance_v1.hh>
namespace cellerator::compiler::diagnostics::v1 {struct explainability_receipt{bool verified_failure=false,trusted_continuation=false,unsafe_native_lowering=false,full_candidate_explanation=false,source_native_trace=false,provenance_stripping=false;};[[nodiscard]] bool freeze_explainability(const explainability_receipt&) noexcept;}
