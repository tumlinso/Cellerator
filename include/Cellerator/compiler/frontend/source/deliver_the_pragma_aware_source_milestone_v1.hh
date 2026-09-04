#pragma once

#include <Cellerator/compiler/frontend/source/construct_shadow_c_placeholders_v1.hh>

#include <string>
#include <vector>

namespace Cellerator::compiler::frontend::source {

struct source_unit_v1 { source_space_id_v1 id = invalid_source_space_v1; std::string path; std::string bytes; };
struct transformed_source_unit_v1 {
    source_space_id_v1 id = invalid_source_space_v1;
    std::string path;
    std::string shadow_bytes;
    bool dialect_activated = false;
    std::vector<shadow_placeholder_v1> placeholders;
};

[[nodiscard]] std::vector<transformed_source_unit_v1> transform_pragma_aware_sources_v1(
    const std::vector<source_unit_v1>& units);

} // namespace Cellerator::compiler::frontend::source
