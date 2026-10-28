#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace cellerator::compiler::pass::v1 {

struct opaque_extension_node_v1 {
    std::uint32_t ir_level = 0;
    std::string qualified_name;
    std::string exact_text;
    std::vector<std::uint8_t> opaque_payload;
};

[[nodiscard]] bool parse_unknown_extension_v1(std::uint32_t ir_level,
    std::string_view qualified_name, std::string_view exact_text,
    opaque_extension_node_v1& output) noexcept;
[[nodiscard]] std::string print_unknown_extension_v1(
    const opaque_extension_node_v1& node);
[[nodiscard]] std::vector<std::uint8_t> serialize_unknown_extension_v1(
    const opaque_extension_node_v1& node);
[[nodiscard]] bool deserialize_unknown_extension_v1(
    const std::vector<std::uint8_t>& bytes, opaque_extension_node_v1& output) noexcept;
[[nodiscard]] opaque_extension_node_v1 clone_unknown_extension_v1(
    const opaque_extension_node_v1& node);
[[nodiscard]] bool forward_unknown_extension_v1(
    const opaque_extension_node_v1& node,
    std::vector<opaque_extension_node_v1>& destination) noexcept;

}  // namespace cellerator::compiler::pass::v1
