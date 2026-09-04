#include <Cellerator/compiler/frontend/source/construct_shadow_c_placeholders_v1.hh>

#include <sstream>

namespace Cellerator::compiler::frontend::source {
namespace {
std::uint64_t stable_id(source_span_v1 span) noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    for (auto value : {std::uint64_t(span.begin.space), span.begin.byte_offset, span.end.byte_offset}) {
        for (unsigned shift = 0; shift != 64; shift += 8) { hash ^= (value >> shift) & 0xffU; hash *= 1099511628211ULL; }
    }
    return hash;
}
}

shadow_cxx_v1 construct_shadow_cxx_v1(source_space_id_v1 source, std::string_view bytes,
                                      const std::vector<source_span_v1>& islands,
                                      const std::vector<std::vector<shadow_capture_slot_v1>>& captures) {
    shadow_cxx_v1 result;
    std::uint64_t cursor = 0;
    for (std::size_t index = 0; index < islands.size(); ++index) {
        const auto span = islands[index];
        if (!span.valid() || span.begin.space != source || span.begin.byte_offset < cursor || span.end.byte_offset > bytes.size()) continue;
        result.bytes.append(bytes.substr(cursor, span.begin.byte_offset - cursor));
        const auto shadow_begin = result.bytes.size();
        const auto id = stable_id(span);
        std::ostringstream replacement;
        replacement << "(cellerator_shadow_field<" << id << "ULL>())";
        result.bytes += replacement.str();
        result.placeholders.push_back({id, span,
            {{source, shadow_begin}, {source, result.bytes.size()}},
            index < captures.size() ? captures[index] : std::vector<shadow_capture_slot_v1>{}});
        cursor = span.end.byte_offset;
    }
    result.bytes.append(bytes.substr(cursor));
    return result;
}

} // namespace Cellerator::compiler::frontend::source
