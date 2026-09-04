#include <Cellerator/compiler/frontend/source/define_macro_expansion_semantics_for_cellerator_tokens_v1.hh>

namespace Cellerator::compiler::frontend::source {

bool macro_construct_is_complete_v1(const std::vector<expanded_cellerator_token_v1>& tokens) noexcept {
    std::uint64_t field_depth = 0;
    for (const auto& token : tokens) {
        if (!macro_token_is_cellerator_v1(token)) return false;
        if (token.role == cellerator_token_role_v1::field_open) ++field_depth;
        if (token.role == cellerator_token_role_v1::field_close) {
            if (field_depth == 0) return false;
            --field_depth;
        }
    }
    return field_depth == 0;
}

} // namespace Cellerator::compiler::frontend::source
