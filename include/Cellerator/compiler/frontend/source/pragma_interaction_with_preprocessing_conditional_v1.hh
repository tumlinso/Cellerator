#pragma once

#include <cstdint>

namespace Cellerator::compiler::frontend::source {

enum class preprocessing_source_v1 : std::uint8_t {
    textual = 1,
    include_replay,
    precompiled_header,
    module,
};

struct pragma_preprocessing_context_v1 {
    bool conditional_path_active = true;
    bool directive_was_skipped = false;
    preprocessing_source_v1 source = preprocessing_source_v1::textual;
    bool event_belongs_to_current_include_instance = true;
};

[[nodiscard]] constexpr bool pragma_may_activate_v1(pragma_preprocessing_context_v1 context) noexcept {
    return context.conditional_path_active && !context.directive_was_skipped &&
           context.event_belongs_to_current_include_instance;
}

} // namespace Cellerator::compiler::frontend::source
