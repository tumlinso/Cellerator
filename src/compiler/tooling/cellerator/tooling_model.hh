#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace cellerator::compiler::tooling::v1 {

struct completion_item {
    std::string spelling;
    std::string category;
};

[[nodiscard]] std::vector<completion_item>
complete_cellerator_syntax(std::string_view source, std::size_t cursor);

struct biological_hover {
    std::string domain, tag, source_axis, destination_axis, support, orientation;
    std::string numeric_tuple, mutability, structure_identity, value_generation, source_link;
};
[[nodiscard]] biological_hover describe_biological_relation(std::string_view declaration);
struct field_effect_view {std::string field,boundary,profile;std::vector<std::string> captures,reads,writes,effects,barriers;bool optimization_visible=false;};
[[nodiscard]] field_effect_view describe_field_effects(std::string_view source,std::size_t cursor);

}  // namespace cellerator::compiler::tooling::v1
