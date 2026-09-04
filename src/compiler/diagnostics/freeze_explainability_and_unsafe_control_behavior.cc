#include <Cellerator/compiler/diagnostics/freeze_explainability_and_unsafe_control_behavior_v1.hh>
namespace cellerator::compiler::diagnostics::v1 {bool freeze_explainability(const explainability_receipt&r) noexcept{return r.verified_failure&&r.trusted_continuation&&r.unsafe_native_lowering&&r.full_candidate_explanation&&r.source_native_trace&&r.provenance_stripping;}}
