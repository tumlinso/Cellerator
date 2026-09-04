#pragma once

#include <Cellerator/compiler/frontend/source/define_the_unified_source_location_model_v1.hh>

#include <cstdint>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::frontend::source {

struct field_island_scan_v1 {
    std::vector<source_span_v1> islands;
    bool balanced = true;
};

[[nodiscard]] field_island_scan_v1 recognize_execution_field_islands_v1(
    source_space_id_v1 source, std::string_view bytes, std::uint64_t activation_offset = 0);

} // namespace Cellerator::compiler::frontend::source
