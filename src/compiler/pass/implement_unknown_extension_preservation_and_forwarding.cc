#include <Cellerator/compiler/pass/implement_unknown_extension_preservation_and_forwarding_v1.hh>

#include <cstring>

namespace cellerator::compiler::pass::v1 {
namespace {
void append_u32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        bytes.push_back(static_cast<std::uint8_t>(value >> shift));
    }
}
bool read_u32(const std::vector<std::uint8_t>& bytes, std::size_t& cursor,
    std::uint32_t& value) {
    if (bytes.size() - cursor < 4) return false;
    value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8) value |=
        static_cast<std::uint32_t>(bytes[cursor++]) << shift;
    return true;
}
bool read_blob(const std::vector<std::uint8_t>& bytes, std::size_t& cursor,
    std::uint32_t size, void* destination) {
    if (size > bytes.size() - cursor) return false;
    if (size != 0) std::memcpy(destination, bytes.data() + cursor, size);
    cursor += size;
    return true;
}
}

bool parse_unknown_extension_v1(std::uint32_t ir_level,
    std::string_view qualified_name, std::string_view exact_text,
    opaque_extension_node_v1& output) noexcept {
    if (qualified_name.empty() || qualified_name.find('.') == std::string_view::npos) {
        return false;
    }
    output = {ir_level, std::string(qualified_name), std::string(exact_text), {}};
    return true;
}

std::string print_unknown_extension_v1(const opaque_extension_node_v1& node) {
    return node.exact_text;
}

std::vector<std::uint8_t> serialize_unknown_extension_v1(
    const opaque_extension_node_v1& node) {
    std::vector<std::uint8_t> bytes;
    append_u32(bytes, node.ir_level);
    append_u32(bytes, static_cast<std::uint32_t>(node.qualified_name.size()));
    append_u32(bytes, static_cast<std::uint32_t>(node.exact_text.size()));
    append_u32(bytes, static_cast<std::uint32_t>(node.opaque_payload.size()));
    bytes.insert(bytes.end(), node.qualified_name.begin(), node.qualified_name.end());
    bytes.insert(bytes.end(), node.exact_text.begin(), node.exact_text.end());
    bytes.insert(bytes.end(), node.opaque_payload.begin(), node.opaque_payload.end());
    return bytes;
}

bool deserialize_unknown_extension_v1(const std::vector<std::uint8_t>& bytes,
    opaque_extension_node_v1& output) noexcept {
    std::size_t cursor = 0;
    std::uint32_t level = 0, name_size = 0, text_size = 0, payload_size = 0;
    if (!read_u32(bytes, cursor, level) || !read_u32(bytes, cursor, name_size)
        || !read_u32(bytes, cursor, text_size) || !read_u32(bytes, cursor, payload_size)) {
        return false;
    }
    opaque_extension_node_v1 parsed;
    parsed.ir_level = level;
    parsed.qualified_name.resize(name_size);
    parsed.exact_text.resize(text_size);
    parsed.opaque_payload.resize(payload_size);
    if (!read_blob(bytes, cursor, name_size, parsed.qualified_name.data())
        || !read_blob(bytes, cursor, text_size, parsed.exact_text.data())
        || !read_blob(bytes, cursor, payload_size, parsed.opaque_payload.data())
        || cursor != bytes.size()) return false;
    output = std::move(parsed);
    return true;
}

opaque_extension_node_v1 clone_unknown_extension_v1(
    const opaque_extension_node_v1& node) { return node; }

bool forward_unknown_extension_v1(const opaque_extension_node_v1& node,
    std::vector<opaque_extension_node_v1>& destination) noexcept {
    try { destination.push_back(node); } catch (...) { return false; }
    return true;
}

}  // namespace cellerator::compiler::pass::v1
