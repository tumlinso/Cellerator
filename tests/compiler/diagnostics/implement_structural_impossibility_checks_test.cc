#include <Cellerator/compiler/diagnostics/implement_structural_impossibility_checks_v1.hh>

#include <array>
#include <cassert>

int main() {
    using namespace cellerator::compiler::diagnostics::v1;
    constexpr std::array failures{
        structural_impossibility::malformed_graph,
        structural_impossibility::missing_required_operand,
        structural_impossibility::impossible_reference,
        structural_impossibility::uninterpretable_text,
        structural_impossibility::unrepresentable_backend_state};
    constexpr std::array modes{
        validation_mode::verified, validation_mode::checked,
        validation_mode::trusted, validation_mode::unsafe,
        validation_mode::unchecked};
    for (const auto mode : modes) {
        for (const auto failure : failures) {
            const auto result = check_structural_possibility(
                {static_cast<std::uint32_t>(failure), mode});
            assert(!result.can_continue);
            assert(result.first_failure == failure);
        }
        assert(check_structural_possibility({0, mode}).can_continue);
    }
}
