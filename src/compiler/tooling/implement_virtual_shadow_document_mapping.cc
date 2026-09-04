#include <Cellerator/compiler/tooling/implement_virtual_shadow_document_mapping_v1.hh>

#include <algorithm>

namespace Cellerator::compiler::tooling {
namespace {
std::optional<std::uint64_t> map_offset(std::uint64_t offset, document_span_v1 from,
                                        document_span_v1 to) noexcept {
    if (!from.valid() || !to.valid() || from.end - from.begin != to.end - to.begin ||
        offset < from.begin || offset > from.end)
        return std::nullopt;
    return to.begin + offset - from.begin;
}
} // namespace

virtual_shadow_document_v1::virtual_shadow_document_v1(std::string original_uri,
                                                       std::string original_text)
    : original_uri_(std::move(original_uri)), original_text_(std::move(original_text)) {}

void virtual_shadow_document_v1::append_generated(std::string_view text) {
    shadow_text_.append(text);
}

bool virtual_shadow_document_v1::append_original(document_span_v1 range) {
    if (!range.valid() || range.end > original_text_.size()) return false;
    const auto shadow_begin = shadow_text_.size();
    shadow_text_.append(original_text_, static_cast<std::size_t>(range.begin),
                        static_cast<std::size_t>(range.end - range.begin));
    segments_.push_back({range, {shadow_begin, shadow_text_.size()}});
    return true;
}

bool virtual_shadow_document_v1::replace_original(document_span_v1 range,
                                                  std::string_view transformed) {
    if (!range.valid() || range.end > original_text_.size() ||
        transformed.size() != range.end - range.begin)
        return false;
    const auto shadow_begin = shadow_text_.size();
    shadow_text_.append(transformed);
    segments_.push_back({range, {shadow_begin, shadow_text_.size()}});
    return true;
}

std::optional<std::uint64_t> virtual_shadow_document_v1::to_original(
    std::uint64_t offset) const noexcept {
    for (const auto &segment : segments_)
        if (const auto mapped = map_offset(offset, segment.shadow, segment.original)) return mapped;
    return std::nullopt;
}

std::optional<std::uint64_t> virtual_shadow_document_v1::to_shadow(
    std::uint64_t offset) const noexcept {
    for (const auto &segment : segments_)
        if (const auto mapped = map_offset(offset, segment.original, segment.shadow)) return mapped;
    return std::nullopt;
}

std::optional<document_span_v1> virtual_shadow_document_v1::to_original(
    document_span_v1 range) const noexcept {
    if (!range.valid()) return std::nullopt;
    for (const auto &segment : segments_) {
        if (range.begin >= segment.shadow.begin && range.end <= segment.shadow.end)
            return document_span_v1{*map_offset(range.begin, segment.shadow, segment.original),
                                    *map_offset(range.end, segment.shadow, segment.original)};
    }
    return std::nullopt;
}

std::optional<document_span_v1> virtual_shadow_document_v1::to_shadow(
    document_span_v1 range) const noexcept {
    if (!range.valid()) return std::nullopt;
    for (const auto &segment : segments_) {
        if (range.begin >= segment.original.begin && range.end <= segment.original.end)
            return document_span_v1{*map_offset(range.begin, segment.original, segment.shadow),
                                    *map_offset(range.end, segment.original, segment.shadow)};
    }
    return std::nullopt;
}

std::optional<mapped_text_edit_v1> virtual_shadow_document_v1::map_edit_to_original(
    const mapped_text_edit_v1 &edit) const {
    const auto mapped = to_original(edit.range);
    if (!mapped) return std::nullopt;
    return mapped_text_edit_v1{*mapped, edit.replacement};
}

} // namespace Cellerator::compiler::tooling
