#include <Cellerator/compiler/frontend/parser/implement_parser_cursor_and_bounded_lookahead_v1.hh>

#include <cassert>
#include <cstdint>
#include <vector>

using namespace Cellerator::compiler::frontend::parser;

int main() {
    std::vector<parser_token_v1> tokens{
        {token_kind::field_open, "<[", 0},
        {token_kind::relation_open, "-[", 3},
        {token_kind::identifier, "r", 5},
        {token_kind::relation_close, "]->", 6},
        {token_kind::field_close, "]>", 10},
        {token_kind::end_of_file, {}, 12},
    };
    parser_cursor_v1 cursor(tokens, 2);
    assert(cursor.peek(2).kind == token_kind::identifier);
    assert(cursor.peek(3).kind == token_kind::end_of_file);
    assert(cursor.error() == cursor_error_v1::lookahead_limit);
    cursor.clear_error();

    const auto origin = cursor.checkpoint();
    for (int index = 0; index != 5; ++index)
        assert(cursor.consume_balanced());
    assert(cursor.delimiter_stack().empty());
    cursor.rollback(origin);
    assert(cursor.position() == 0);

    std::uint32_t state = 0x31415926u;
    for (int trial = 0; trial != 256; ++trial) {
        const auto saved = cursor.checkpoint();
        state = state * 1664525u + 1013904223u;
        const auto steps = static_cast<std::size_t>(state % 5u);
        for (std::size_t step = 0; step < steps; ++step)
            cursor.consume_balanced();
        cursor.rollback(saved);
        assert(cursor.position() == saved.position);
        assert(cursor.delimiter_stack() == saved.delimiter_stack);
    }

    parser_cursor_v1 mismatch({
        {token_kind::relation_close, "]->", 0},
        {token_kind::kw_verify, "verify", 4},
    });
    assert(!mismatch.consume_balanced());
    assert(mismatch.error() == cursor_error_v1::mismatched_delimiter);
    assert(mismatch.recover_to({token_kind::kw_verify}));
    assert(mismatch.peek().kind == token_kind::kw_verify);
}
