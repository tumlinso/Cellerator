#include <Cellerator/compiler/frontend/source/build_a_lossless_raw_token_stream_v1.hh>

#include <cctype>

namespace Cellerator::compiler::frontend::source {

raw_token_stream_v1 build_raw_token_stream_v1(source_space_id_v1 source, std::string_view bytes,
                                               std::uint64_t activation_offset,
                                               std::uint64_t preprocessor_condition) {
    raw_token_stream_v1 stream{};
    stream.source = source;
    std::size_t cursor = 0;
    while (cursor < bytes.size()) {
        const auto trivia_begin = cursor;
        while (cursor < bytes.size() && std::isspace(static_cast<unsigned char>(bytes[cursor]))) {
            ++cursor;
        }
        if (cursor == bytes.size()) {
            stream.trailing_trivia.assign(bytes.substr(trivia_begin));
            break;
        }
        const auto token_begin = cursor;
        const auto first = static_cast<unsigned char>(bytes[cursor]);
        if (std::isalnum(first) || bytes[cursor] == '_') {
            while (cursor < bytes.size()) {
                const auto value = static_cast<unsigned char>(bytes[cursor]);
                if (!std::isalnum(value) && bytes[cursor] != '_') break;
                ++cursor;
            }
        } else {
            ++cursor;
        }
        stream.tokens.push_back({std::string(bytes.substr(trivia_begin, token_begin - trivia_begin)),
                                 std::string(bytes.substr(token_begin, cursor - token_begin)),
                                 {{source, token_begin}, {source, cursor}}, std::nullopt,
                                 token_begin >= activation_offset, preprocessor_condition});
    }
    return stream;
}

std::string reconstruct_raw_token_stream_v1(const raw_token_stream_v1& stream) {
    std::string result;
    for (const auto& token : stream.tokens) {
        result += token.leading_trivia;
        result += token.spelling;
    }
    result += stream.trailing_trivia;
    return result;
}

bool has_exact_byte_coverage_v1(const raw_token_stream_v1& stream,
                                std::uint64_t source_size) noexcept {
    std::uint64_t cursor = 0;
    for (const auto& token : stream.tokens) {
        cursor += token.leading_trivia.size();
        if (!token.span.valid() || token.span.begin.space != stream.source ||
            token.span.begin.byte_offset != cursor || token.span.size_bytes() != token.spelling.size()) {
            return false;
        }
        cursor = token.span.end.byte_offset;
    }
    return cursor + stream.trailing_trivia.size() == source_size;
}

} // namespace Cellerator::compiler::frontend::source
