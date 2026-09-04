#pragma once

#include <Cellerator/compiler/frontend/source/define_the_unified_source_location_model_v1.hh>

#include <cstdint>
#include <string>
#include <string_view>

namespace Cellerator::compiler::frontend::source {

enum class pragma_diagnostic_v1 : std::uint8_t {
    none = 0,
    duplicate,
    late_activation,
    malformed,
    macro_produced,
    unknown_version,
};

struct pragma_request_v1 {
    std::string_view payload;
    source_location_v1 location{};
    bool already_active = false;
    bool saw_non_directive_token = false;
    bool produced_by_macro = false;
};

struct pragma_result_v1 {
    bool activate = false;
    std::string revision;
    source_location_v1 location{};
    pragma_diagnostic_v1 diagnostic = pragma_diagnostic_v1::none;
};

[[nodiscard]] pragma_result_v1 handle_cellerator_pragma_v1(pragma_request_v1 request);

} // namespace Cellerator::compiler::frontend::source
