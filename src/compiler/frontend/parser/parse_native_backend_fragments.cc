#include <Cellerator/compiler/frontend/parser/parse_native_backend_fragments_v1.hh>

#include <cctype>
#include <optional>

namespace Cellerator::compiler::frontend::parser {
namespace {

std::string trim_copy(std::string_view value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())))
        value.remove_prefix(1);
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())))
        value.remove_suffix(1);
    return std::string(value);
}

std::size_t matching_body(std::string_view source, std::size_t open) {
    unsigned depth = 0;
    bool quoted = false;
    bool character = false;
    bool line_comment = false;
    bool block_comment = false;
    for (auto offset = open; offset < source.size(); ++offset) {
        const auto next = offset + 1 < source.size() ? source[offset + 1] : '\0';
        if (line_comment) {
            line_comment = source[offset] != '\n';
            continue;
        }
        if (block_comment) {
            if (source[offset] == '*' && next == '/') {
                block_comment = false;
                ++offset;
            }
            continue;
        }
        if (!quoted && !character && source[offset] == '/' && next == '/') {
            line_comment = true;
            ++offset;
            continue;
        }
        if (!quoted && !character && source[offset] == '/' && next == '*') {
            block_comment = true;
            ++offset;
            continue;
        }
        const bool escaped = offset != 0 && source[offset - 1] == '\\';
        if (!character && source[offset] == '"' && !escaped)
            quoted = !quoted;
        else if (!quoted && source[offset] == '\'' && !escaped)
            character = !character;
        if (quoted || character)
            continue;
        if (source[offset] == '{')
            ++depth;
        else if (source[offset] == '}' && --depth == 0)
            return offset;
    }
    return std::string_view::npos;
}

std::optional<native_backend_kind_v1> backend_of(std::string_view name) {
    if (name == "generated_cxx") return native_backend_kind_v1::generated_cxx;
    if (name == "cuda") return native_backend_kind_v1::cuda;
    if (name == "ptx") return native_backend_kind_v1::ptx;
    if (name == "raw_native") return native_backend_kind_v1::raw_native;
    return std::nullopt;
}

std::string_view clause(std::string_view header, std::string_view name) {
    const auto marker = std::string(name) + "(";
    const auto begin = header.find(marker);
    if (begin == std::string_view::npos)
        return {};
    const auto contents = begin + marker.size();
    unsigned depth = 1;
    for (auto offset = contents; offset < header.size(); ++offset) {
        depth += header[offset] == '(' ? 1u : 0u;
        if (header[offset] == ')' && --depth == 0)
            return header.substr(contents, offset - contents);
    }
    return {};
}

std::vector<std::string> comma_list(std::string_view value) {
    std::vector<std::string> result;
    std::size_t begin = 0;
    unsigned depth = 0;
    for (std::size_t offset = 0; offset < value.size(); ++offset) {
        depth += value[offset] == '(' || value[offset] == '<' ? 1u : 0u;
        if ((value[offset] == ')' || value[offset] == '>') && depth)
            --depth;
        if (value[offset] == ',' && depth == 0) {
            result.push_back(trim_copy(value.substr(begin, offset - begin)));
            begin = offset + 1;
        }
    }
    const auto tail = trim_copy(value.substr(begin));
    if (!tail.empty())
        result.push_back(tail);
    return result;
}

} // namespace

native_backend_fragment_parse_v1 parse_native_backend_fragments_v1(
    std::string_view source) {
    native_backend_fragment_parse_v1 result;
    std::size_t search = 0;
    while ((search = source.find("native<", search)) != std::string_view::npos) {
        const auto backend_end = source.find('>', search + 7);
        if (backend_end == std::string_view::npos) {
            result.diagnostics.push_back({"unterminated native backend kind",
                                          {search, source.size()}});
            break;
        }
        const auto backend = backend_of(source.substr(search + 7, backend_end - search - 7));
        if (!backend) {
            result.diagnostics.push_back({"unknown native backend kind",
                                          {search, backend_end + 1}});
            search = backend_end + 1;
            continue;
        }
        const auto body_open = source.find('{', backend_end + 1);
        if (body_open == std::string_view::npos) {
            result.diagnostics.push_back({"native fragment has no payload",
                                          {search, source.size()}});
            break;
        }
        const auto body_close = matching_body(source, body_open);
        if (body_close == std::string_view::npos) {
            result.diagnostics.push_back({"unterminated native fragment payload",
                                          {body_open, source.size()}});
            break;
        }

        native_backend_fragment_v1 fragment;
        fragment.backend = *backend;
        fragment.range = {search, body_close + 1};
        fragment.spelling = std::string(source.substr(search, body_close - search + 1));
        fragment.payload = std::string(source.substr(body_open + 1,
                                                     body_close - body_open - 1));
        const auto header = source.substr(backend_end + 1, body_open - backend_end - 1);
        fragment.target = trim_copy(clause(header, "target"));
        fragment.inputs = comma_list(clause(header, "inputs"));
        fragment.outputs = comma_list(clause(header, "outputs"));
        fragment.clobbers = comma_list(clause(header, "clobbers"));
        fragment.effects = comma_list(clause(header, "effects"));
        fragment.fallback = trim_copy(clause(header, "fallback"));

        if (fragment.target.empty())
            result.diagnostics.push_back({"native fragment requires an explicit target",
                                          fragment.range});
        if (fragment.inputs.empty())
            result.diagnostics.push_back({"native fragment requires explicit inputs",
                                          fragment.range});
        if (fragment.outputs.empty())
            result.diagnostics.push_back({"native fragment requires explicit outputs",
                                          fragment.range});
        if (fragment.clobbers.empty() && fragment.effects.empty())
            result.diagnostics.push_back({"native fragment requires clobbers or effects",
                                          fragment.range});
        if (fragment.fallback.empty())
            result.diagnostics.push_back({"native fragment requires an exact fallback",
                                          fragment.range});
        result.fragments.push_back(std::move(fragment));
        search = body_close + 1;
    }
    return result;
}

} // namespace Cellerator::compiler::frontend::parser
