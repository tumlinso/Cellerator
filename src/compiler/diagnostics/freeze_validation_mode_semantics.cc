#include <Cellerator/compiler/diagnostics/freeze_validation_mode_semantics_v1.hh>

namespace cellerator::compiler::diagnostics::v1 {

validation_failure_policy validation_policy(validation_mode mode,
                                            validation_domain domain) noexcept {
    switch (domain) {
    case validation_domain::parsing:
    case validation_domain::exact_coverage:
    case validation_domain::resources:
    case validation_domain::backend_support:
        return validation_failure_policy::mandatory_failure;
    case validation_domain::semantic_invariants:
        if (mode == validation_mode::unsafe) {
            return validation_failure_policy::overridable_diagnostic;
        }
        if (mode == validation_mode::unchecked) {
            return validation_failure_policy::skipped;
        }
        return validation_failure_policy::mandatory_failure;
    case validation_domain::numerical_claims:
        if (mode == validation_mode::verified) {
            return validation_failure_policy::mandatory_failure;
        }
        if (mode == validation_mode::checked) {
            return validation_failure_policy::overridable_diagnostic;
        }
        if (mode == validation_mode::trusted) {
            return validation_failure_policy::accepted_trust_assertion;
        }
        return validation_failure_policy::skipped;
    case validation_domain::native_bindings:
        if (mode == validation_mode::verified ||
            mode == validation_mode::checked) {
            return validation_failure_policy::mandatory_failure;
        }
        if (mode == validation_mode::trusted) {
            return validation_failure_policy::accepted_trust_assertion;
        }
        if (mode == validation_mode::unsafe) {
            return validation_failure_policy::overridable_diagnostic;
        }
        return validation_failure_policy::skipped;
    }
    return validation_failure_policy::mandatory_failure;
}

} // namespace cellerator::compiler::diagnostics::v1
