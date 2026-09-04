#pragma once

#include <cstdint>
#include <string_view>

namespace Cellerator::compiler::frontend::source {

enum class source_input_mode_v1 : std::uint8_t {
    ordinary_cxx = 1,
    activated_cellerator,
    standalone_ceir,
};

struct source_input_classification_v1 {
    source_input_mode_v1 mode = source_input_mode_v1::ordinary_cxx;
    std::uint64_t activation_offset = 0;
    std::string_view revision{};
};

[[nodiscard]] source_input_classification_v1 classify_source_input_v1(
    std::string_view path, std::string_view bytes) noexcept;

} // namespace Cellerator::compiler::frontend::source
