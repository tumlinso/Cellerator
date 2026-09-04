#include <Cellerator/compiler/diagnostics/implement_advisory_semantic_validators_v1.hh>
namespace cellerator::compiler::diagnostics::v1 {
advisory_disposition validate_semantic_advisory(const advisory_request&r) noexcept {
 if(!r.representable)return advisory_disposition::error;
 if(r.mode==validation_mode::verified||r.escalate_warning)return r.force_continuation?advisory_disposition::error:advisory_disposition::error;
 if(r.mode==validation_mode::trusted&&r.force_continuation)return advisory_disposition::trusted_continuation;
 if(r.mode==validation_mode::unsafe&&r.force_continuation)return advisory_disposition::unsafe_continuation;
 if(r.suppress_warning&&(r.mode==validation_mode::unsafe||r.mode==validation_mode::unchecked))return advisory_disposition::suppressed;
 return advisory_disposition::warning;
}
}
