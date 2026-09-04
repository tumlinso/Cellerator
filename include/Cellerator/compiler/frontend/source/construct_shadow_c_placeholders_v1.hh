#pragma once

#include <Cellerator/compiler/frontend/source/define_the_unified_source_location_model_v1.hh>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::frontend::source {

struct shadow_capture_slot_v1 { std::string name; std::string type_spelling; };
struct shadow_placeholder_v1 {
    std::uint64_t stable_id = 0;
    source_span_v1 original{};
    source_span_v1 shadow{};
    std::vector<shadow_capture_slot_v1> captures;
};
struct shadow_cxx_v1 { std::string bytes; std::vector<shadow_placeholder_v1> placeholders; };

[[nodiscard]] shadow_cxx_v1 construct_shadow_cxx_v1(
    source_space_id_v1 source, std::string_view bytes, const std::vector<source_span_v1>& islands,
    const std::vector<std::vector<shadow_capture_slot_v1>>& captures = {});

} // namespace Cellerator::compiler::frontend::source
