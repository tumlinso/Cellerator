#pragma once

#include <Cellerator/compiler/frontend/source/define_the_unified_source_location_model_v1.hh>

#include <cstdint>
#include <vector>

namespace Cellerator::compiler::frontend::source {

enum class cellerator_token_role_v1 : std::uint8_t {
    identifier = 1,
    field_open,
    field_close,
    relation_arrow,
    attribute,
};

struct expanded_cellerator_token_v1 {
    cellerator_token_role_v1 role = cellerator_token_role_v1::identifier;
    source_span_v1 spelling{};
    source_span_v1 expansion{};
    bool expansion_dialect_active = false;
};

[[nodiscard]] constexpr bool macro_token_is_cellerator_v1(
    const expanded_cellerator_token_v1& token) noexcept {
    return token.spelling.valid() && token.expansion.valid() && token.expansion_dialect_active;
}
[[nodiscard]] bool macro_construct_is_complete_v1(
    const std::vector<expanded_cellerator_token_v1>& tokens) noexcept;

} // namespace Cellerator::compiler::frontend::source
