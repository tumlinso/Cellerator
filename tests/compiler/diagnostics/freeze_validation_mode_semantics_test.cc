#include <Cellerator/compiler/diagnostics/freeze_validation_mode_semantics_v1.hh>

#include <cassert>
#include <array>

int main() {
    using namespace cellerator::compiler::diagnostics::v1;
    for (const auto mode : std::array{
             validation_mode::verified, validation_mode::checked,
             validation_mode::trusted, validation_mode::unsafe,
             validation_mode::unchecked}) {
        assert(validation_policy(mode, validation_domain::parsing) ==
               validation_failure_policy::mandatory_failure);
        assert(validation_policy(mode, validation_domain::exact_coverage) ==
               validation_failure_policy::mandatory_failure);
        assert(validation_policy(mode, validation_domain::resources) ==
               validation_failure_policy::mandatory_failure);
        assert(validation_policy(mode, validation_domain::backend_support) ==
               validation_failure_policy::mandatory_failure);
    }
    assert(validation_policy(validation_mode::verified,
                             validation_domain::numerical_claims) ==
           validation_failure_policy::mandatory_failure);
    assert(validation_policy(validation_mode::trusted,
                             validation_domain::numerical_claims) ==
           validation_failure_policy::accepted_trust_assertion);
    assert(validation_policy(validation_mode::unsafe,
                             validation_domain::semantic_invariants) ==
           validation_failure_policy::overridable_diagnostic);
    assert(validation_policy(validation_mode::unchecked,
                             validation_domain::native_bindings) ==
           validation_failure_policy::skipped);
}
