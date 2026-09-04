#include <Cellerator/compiler/diagnostics/implement_structural_impossibility_checks_v1.hh>

#include <array>

namespace cellerator::compiler::diagnostics::v1 {

structural_check_result check_structural_possibility(
    const structural_check_request& request) noexcept {
    (void)request.mode; // Structural impossibility is mode-independent.
    for (const auto failure : std::array{
             structural_impossibility::malformed_graph,
             structural_impossibility::missing_required_operand,
             structural_impossibility::impossible_reference,
             structural_impossibility::uninterpretable_text,
             structural_impossibility::unrepresentable_backend_state}) {
        if ((request.failures & static_cast<std::uint32_t>(failure)) != 0U) {
            return {false, failure};
        }
    }
    return {true, structural_impossibility::none};
}

} // namespace cellerator::compiler::diagnostics::v1
