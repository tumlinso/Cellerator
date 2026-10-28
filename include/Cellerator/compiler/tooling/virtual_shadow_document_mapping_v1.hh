#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::tooling {

struct document_span_v1 {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;

    [[nodiscard]] constexpr bool valid() const noexcept { return begin <= end; }
};

struct shadow_mapping_segment_v1 {
    document_span_v1 original;
    document_span_v1 shadow;
};

struct mapped_text_edit_v1 {
    document_span_v1 range;
    std::string replacement;
};

class virtual_shadow_document_v1 {
public:
    explicit virtual_shadow_document_v1(std::string original_uri, std::string original_text);

    void append_generated(std::string_view text);
    [[nodiscard]] bool append_original(document_span_v1 range);
    [[nodiscard]] bool replace_original(document_span_v1 range, std::string_view transformed);

    [[nodiscard]] std::optional<std::uint64_t> to_original(std::uint64_t shadow_offset) const noexcept;
    [[nodiscard]] std::optional<std::uint64_t> to_shadow(std::uint64_t original_offset) const noexcept;
    [[nodiscard]] std::optional<document_span_v1> to_original(document_span_v1 shadow_range) const noexcept;
    [[nodiscard]] std::optional<document_span_v1> to_shadow(document_span_v1 original_range) const noexcept;
    [[nodiscard]] std::optional<mapped_text_edit_v1> map_edit_to_original(
        const mapped_text_edit_v1 &edit) const;

    [[nodiscard]] const std::string &original_uri() const noexcept { return original_uri_; }
    [[nodiscard]] const std::string &original_text() const noexcept { return original_text_; }
    [[nodiscard]] const std::string &shadow_text() const noexcept { return shadow_text_; }
    [[nodiscard]] const std::vector<shadow_mapping_segment_v1> &segments() const noexcept {
        return segments_;
    }

private:
    std::string original_uri_;
    std::string original_text_;
    std::string shadow_text_;
    std::vector<shadow_mapping_segment_v1> segments_;
};

} // namespace Cellerator::compiler::tooling
