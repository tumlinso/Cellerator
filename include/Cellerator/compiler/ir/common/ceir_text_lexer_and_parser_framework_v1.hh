#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace cellerator::compiler::ir::text {

enum class token_kind { word, string_literal, punctuation, end, invalid };
struct byte_range { std::size_t begin{}; std::size_t end{}; };
struct token { token_kind kind{token_kind::invalid}; std::string_view text; byte_range range; };
struct diagnostic { byte_range range; std::string message; };
struct parsed_operation { std::string qualified_name; byte_range range; bool inline_block{}; };
struct parsed_unit {
    std::vector<std::string> includes;
    std::vector<std::string> imports;
    std::vector<parsed_operation> operations;
    std::vector<diagnostic> diagnostics;
};

std::vector<token> lex(std::string_view source);

class parser {
public:
    using dialect_callback = std::function<bool(std::string_view)>;
    void register_dialect(std::string name, dialect_callback callback);
    parsed_unit parse(std::string_view source) const;
private:
    std::unordered_map<std::string, dialect_callback> dialects_;
};

} // namespace cellerator::compiler::ir::text
