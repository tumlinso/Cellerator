#include <Cellerator/compiler/ast/create_structured_frontend_diagnostic_records_v1.hh>

#include <sstream>

namespace Cellerator::compiler::ast {
namespace {

void append_u64(std::string& bytes, std::uint64_t value) {
    for (unsigned shift = 0; shift != 64; shift += 8) bytes.push_back(char(value >> shift));
}

void append_string(std::string& bytes, std::string_view value) {
    append_u64(bytes, value.size());
    bytes.append(value);
}

void append_span(std::string& bytes, frontend::source::source_span_v1 span) {
    append_u64(bytes, span.begin.space);
    append_u64(bytes, span.begin.byte_offset);
    append_u64(bytes, span.end.byte_offset);
}

struct reader_v1 {
    std::string_view bytes;
    std::size_t cursor = 0;
    bool u64(std::uint64_t* value) {
        if (bytes.size() - cursor < 8) return false;
        *value = 0;
        for (unsigned shift = 0; shift != 64; shift += 8)
            *value |= std::uint64_t(std::uint8_t(bytes[cursor++])) << shift;
        return true;
    }
    bool string(std::string* value) {
        std::uint64_t size = 0;
        if (!u64(&size) || size > bytes.size() - cursor) return false;
        value->assign(bytes.substr(cursor, size));
        cursor += size;
        return true;
    }
    bool span(frontend::source::source_span_v1* value) {
        std::uint64_t space = 0, begin = 0, end = 0;
        if (!u64(&space) || !u64(&begin) || !u64(&end) || space > UINT32_MAX) return false;
        *value = {{static_cast<std::uint32_t>(space), begin},
                  {static_cast<std::uint32_t>(space), end}};
        return value->valid();
    }
};

std::string json_escape(std::string_view input) {
    std::string result;
    for (const char byte : input) {
        if (byte == '"' || byte == '\\') result.push_back('\\');
        if (byte == '\n') result += "\\n";
        else result.push_back(byte);
    }
    return result;
}

} // namespace

bool validate_frontend_diagnostic_v1(const frontend_diagnostic_v1& diagnostic,
                                     std::string* error) {
    const auto fail = [&](std::string message) {
        if (error) *error = std::move(message);
        return false;
    };
    if (diagnostic.stable_id == 0 || diagnostic.message.empty())
        return fail("diagnostic requires a stable id and message");
    if (static_cast<unsigned>(diagnostic.severity) < 1 ||
        static_cast<unsigned>(diagnostic.severity) > 4 ||
        static_cast<unsigned>(diagnostic.category) < 1 ||
        static_cast<unsigned>(diagnostic.category) > 6 ||
        static_cast<unsigned>(diagnostic.phase) < 1 ||
        static_cast<unsigned>(diagnostic.phase) > 6)
        return fail("diagnostic classification is invalid");
    if (diagnostic.source_ranges.empty()) return fail("diagnostic requires a source range");
    for (const auto& range : diagnostic.source_ranges)
        if (!range.valid()) return fail("diagnostic source range is invalid");
    for (const auto& note : diagnostic.notes)
        if (note.message.empty() || (note.source && !note.source->valid()))
            return fail("diagnostic note is invalid");
    for (const auto& fix : diagnostic.fix_its)
        if (!fix.source.valid()) return fail("diagnostic fix-it range is invalid");
    if (error) error->clear();
    return true;
}

std::string serialize_frontend_diagnostic_v1(const frontend_diagnostic_v1& diagnostic) {
    std::string bytes{"CEDIAG01", 8};
    append_u64(bytes, diagnostic.stable_id);
    bytes.push_back(char(diagnostic.severity));
    bytes.push_back(char(diagnostic.category));
    bytes.push_back(char(diagnostic.phase));
    append_string(bytes, diagnostic.message);
    append_u64(bytes, diagnostic.source_ranges.size());
    for (const auto range : diagnostic.source_ranges) append_span(bytes, range);
    append_u64(bytes, diagnostic.notes.size());
    for (const auto& note : diagnostic.notes) {
        append_string(bytes, note.message);
        bytes.push_back(note.source ? 1 : 0);
        if (note.source) append_span(bytes, *note.source);
    }
    append_u64(bytes, diagnostic.fix_its.size());
    for (const auto& fix : diagnostic.fix_its) {
        append_span(bytes, fix.source);
        append_string(bytes, fix.replacement);
    }
    append_u64(bytes, diagnostic.related_symbols.size());
    for (const auto symbol : diagnostic.related_symbols) append_u64(bytes, symbol);
    return bytes;
}

std::optional<frontend_diagnostic_v1>
deserialize_frontend_diagnostic_v1(std::string_view bytes, std::string* error) {
    auto fail = [&](std::string message) -> std::optional<frontend_diagnostic_v1> {
        if (error) *error = std::move(message);
        return std::nullopt;
    };
    if (bytes.substr(0, 8) != "CEDIAG01") return fail("invalid diagnostic schema");
    reader_v1 reader{bytes, 8};
    frontend_diagnostic_v1 result;
    std::uint64_t count = 0;
    if (!reader.u64(&result.stable_id) || reader.bytes.size() - reader.cursor < 3)
        return fail("truncated diagnostic header");
    result.severity = diagnostic_severity_v1(std::uint8_t(reader.bytes[reader.cursor++]));
    result.category = diagnostic_category_v1(std::uint8_t(reader.bytes[reader.cursor++]));
    result.phase = compiler_phase_v1(std::uint8_t(reader.bytes[reader.cursor++]));
    if (!reader.string(&result.message) || !reader.u64(&count) || count > 1'000'000)
        return fail("invalid diagnostic body");
    result.source_ranges.resize(count);
    for (auto& range : result.source_ranges) if (!reader.span(&range)) return fail("invalid range");
    if (!reader.u64(&count) || count > 1'000'000) return fail("invalid note count");
    result.notes.resize(count);
    for (auto& note : result.notes) {
        if (!reader.string(&note.message) || reader.cursor == reader.bytes.size())
            return fail("invalid note");
        if (reader.bytes[reader.cursor++]) {
            frontend::source::source_span_v1 span;
            if (!reader.span(&span)) return fail("invalid note range");
            note.source = span;
        }
    }
    if (!reader.u64(&count) || count > 1'000'000) return fail("invalid fix-it count");
    result.fix_its.resize(count);
    for (auto& fix : result.fix_its)
        if (!reader.span(&fix.source) || !reader.string(&fix.replacement)) return fail("invalid fix-it");
    if (!reader.u64(&count) || count > 1'000'000) return fail("invalid symbol count");
    result.related_symbols.resize(count);
    for (auto& symbol : result.related_symbols) if (!reader.u64(&symbol)) return fail("invalid symbol");
    if (reader.cursor != reader.bytes.size() || !validate_frontend_diagnostic_v1(result, error))
        return std::nullopt;
    return result;
}

std::string render_terminal_diagnostic_v1(const frontend_diagnostic_v1& diagnostic) {
    static constexpr const char* names[] = {"", "note", "warning", "error", "fatal"};
    const auto& range = diagnostic.source_ranges.front();
    std::ostringstream out;
    out << names[static_cast<unsigned>(diagnostic.severity)] << "[CE" << diagnostic.stable_id
        << "]: " << diagnostic.message << " @" << range.begin.space << ':'
        << range.begin.byte_offset << '-' << range.end.byte_offset;
    for (const auto& note : diagnostic.notes) out << "\n  note: " << note.message;
    return out.str();
}

std::string render_lsp_diagnostic_v1(const frontend_diagnostic_v1& diagnostic) {
    const auto& range = diagnostic.source_ranges.front();
    std::ostringstream out;
    out << "{\"code\":\"CE" << diagnostic.stable_id << "\",\"severity\":"
        << static_cast<unsigned>(diagnostic.severity) << ",\"message\":\""
        << json_escape(diagnostic.message) << "\",\"range\":{\"space\":"
        << range.begin.space << ",\"begin\":" << range.begin.byte_offset
        << ",\"end\":" << range.end.byte_offset << "},\"data\":{\"category\":"
        << static_cast<unsigned>(diagnostic.category) << ",\"phase\":"
        << static_cast<unsigned>(diagnostic.phase) << ",\"notes\":"
        << diagnostic.notes.size() << ",\"fixIts\":" << diagnostic.fix_its.size()
        << ",\"relatedSymbols\":" << diagnostic.related_symbols.size() << "}}";
    return out.str();
}

} // namespace Cellerator::compiler::ast
