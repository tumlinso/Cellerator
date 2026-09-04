#pragma once

#include <Cellerator/compiler/frontend/parser/freeze_the_executable_grammar_revision_and_token_vocabul_v1.hh>

#include <cstddef>
#include <initializer_list>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::frontend::parser {

struct parser_token_v1 {
    token_kind kind = token_kind::end_of_file;
    std::string_view spelling{};
    std::size_t source_offset = 0;
};

struct cursor_checkpoint_v1 {
    std::size_t position = 0;
    std::vector<token_kind> delimiter_stack;
};

enum class cursor_error_v1 {
    none,
    lookahead_limit,
    mismatched_delimiter,
    unexpected_end
};

class parser_cursor_v1 {
public:
    explicit parser_cursor_v1(std::vector<parser_token_v1> tokens,
                              std::size_t maximum_lookahead = 8);

    [[nodiscard]] const parser_token_v1 &peek(std::size_t distance = 0) noexcept;
    [[nodiscard]] parser_token_v1 consume() noexcept;
    [[nodiscard]] cursor_checkpoint_v1 checkpoint() const;
    void rollback(const cursor_checkpoint_v1 &checkpoint);
    bool consume_balanced() noexcept;
    bool recover_to(std::initializer_list<token_kind> synchronizers) noexcept;

    [[nodiscard]] std::size_t position() const noexcept { return position_; }
    [[nodiscard]] bool at_end() const noexcept;
    [[nodiscard]] cursor_error_v1 error() const noexcept { return error_; }
    void clear_error() noexcept { error_ = cursor_error_v1::none; }
    [[nodiscard]] const std::vector<token_kind> &delimiter_stack() const noexcept {
        return delimiters_;
    }

private:
    std::vector<parser_token_v1> tokens_;
    std::size_t position_ = 0;
    std::size_t maximum_lookahead_ = 0;
    std::vector<token_kind> delimiters_;
    cursor_error_v1 error_ = cursor_error_v1::none;
};

} // namespace Cellerator::compiler::frontend::parser
