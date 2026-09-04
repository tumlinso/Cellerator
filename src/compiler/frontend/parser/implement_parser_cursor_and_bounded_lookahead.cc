#include <Cellerator/compiler/frontend/parser/implement_parser_cursor_and_bounded_lookahead_v1.hh>

#include <algorithm>
#include <utility>

namespace Cellerator::compiler::frontend::parser {
namespace {

const parser_token_v1 end_token{};

bool is_open(token_kind kind) noexcept {
    return kind == token_kind::field_open || kind == token_kind::relation_open;
}

token_kind matching_open(token_kind kind) noexcept {
    return kind == token_kind::field_close ? token_kind::field_open
                                           : token_kind::relation_open;
}

bool is_close(token_kind kind) noexcept {
    return kind == token_kind::field_close || kind == token_kind::relation_close;
}

} // namespace

parser_cursor_v1::parser_cursor_v1(std::vector<parser_token_v1> tokens,
                                   std::size_t maximum_lookahead)
    : tokens_(std::move(tokens)), maximum_lookahead_(maximum_lookahead) {}

const parser_token_v1 &parser_cursor_v1::peek(std::size_t distance) noexcept {
    if (distance > maximum_lookahead_) {
        error_ = cursor_error_v1::lookahead_limit;
        return end_token;
    }
    const auto target = position_ + distance;
    return target < tokens_.size() ? tokens_[target] : end_token;
}

parser_token_v1 parser_cursor_v1::consume() noexcept {
    if (position_ >= tokens_.size()) {
        error_ = cursor_error_v1::unexpected_end;
        return end_token;
    }
    return tokens_[position_++];
}

cursor_checkpoint_v1 parser_cursor_v1::checkpoint() const {
    return {position_, delimiters_};
}

void parser_cursor_v1::rollback(const cursor_checkpoint_v1 &saved) {
    position_ = std::min(saved.position, tokens_.size());
    delimiters_ = saved.delimiter_stack;
    error_ = cursor_error_v1::none;
}

bool parser_cursor_v1::consume_balanced() noexcept {
    const auto token = consume();
    if (token.kind == token_kind::end_of_file)
        return false;
    if (is_open(token.kind)) {
        delimiters_.push_back(token.kind);
        return true;
    }
    if (!is_close(token.kind))
        return true;
    if (delimiters_.empty() || delimiters_.back() != matching_open(token.kind)) {
        error_ = cursor_error_v1::mismatched_delimiter;
        return false;
    }
    delimiters_.pop_back();
    return true;
}

bool parser_cursor_v1::recover_to(
    std::initializer_list<token_kind> synchronizers) noexcept {
    delimiters_.clear();
    while (!at_end()) {
        const auto kind = tokens_[position_].kind;
        if (std::find(synchronizers.begin(), synchronizers.end(), kind)
            != synchronizers.end()) {
            error_ = cursor_error_v1::none;
            return true;
        }
        ++position_;
    }
    error_ = cursor_error_v1::unexpected_end;
    return false;
}

bool parser_cursor_v1::at_end() const noexcept {
    return position_ >= tokens_.size()
        || tokens_[position_].kind == token_kind::end_of_file;
}

} // namespace Cellerator::compiler::frontend::parser
