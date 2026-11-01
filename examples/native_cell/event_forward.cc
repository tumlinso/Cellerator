#include <Cellerator/cells/event.hh>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Token = std::uint32_t;
using EventCell = cellerator::cells::EventCell<Token, std::string>;
using EventView = cellerator::cells::EventView<Token>;

// The owner publishes vector growth through one stable descriptor. Production
// stores can use their own capacity and invalidation policy.
class EventBuffer {
public:
    explicit EventBuffer(std::vector<Token> initial, std::string identity)
        : tokens_(std::move(initial)), identity_(std::move(identity)) {
        refresh_descriptor();
    }

    EventBuffer(const EventBuffer&) = delete;
    EventBuffer& operator=(const EventBuffer&) = delete;
    EventBuffer(EventBuffer&&) = delete;
    EventBuffer& operator=(EventBuffer&&) = delete;

    EventCell cell() const {
        return EventCell(descriptor_, 0, tokens_.size(), identity_);
    }

    void require_current_tail(const EventCell& prefix) const {
        if (&prefix.buffer_descriptor() != &descriptor_) {
            throw std::logic_error("cannot append from a prefix owned by another buffer");
        }
        if (prefix.identity() != identity_ || prefix.offset() != 0 ||
            prefix.limit() != tokens_.size()) {
            throw std::logic_error(
                "cannot append from a prefix that already has a continuation");
        }
    }

    EventCell append_continuation(const EventCell& prefix, Token token) {
        require_current_tail(prefix);

        // vector growth is amortized; copying the full history on every step
        // would make constructing a trajectory quadratic in its length.
        tokens_.push_back(token);
        refresh_descriptor();
        ++append_count_;

        return EventCell(descriptor_, prefix.offset(), prefix.limit() + 1,
                         prefix.identity());
    }

    std::size_t size() const noexcept { return tokens_.size(); }
    std::size_t append_count() const noexcept { return append_count_; }

private:
    void refresh_descriptor() noexcept {
        descriptor_ = {tokens_.data(), tokens_.size(),
                       {cellerator::memory::domain::host, -1, -1, 0}};
    }

    std::vector<Token> tokens_;
    std::string identity_;
    cellerator::memory::const_array_view<Token> descriptor_{};
    std::size_t append_count_ = 0;
};

// A deterministic interface example only. It is neither a trained model nor
// a scientific Karyoplast predictor.
struct DeterministicEvaluator {
    std::size_t& calls;

    Token operator()(EventView input) const {
        ++calls;
        if (input.tokens.count == 0) return 1;
        const Token last = input.tokens.data[input.tokens.count - 1];
        return (last * 3u + 1u) % 17u;
    }
};

// The operation reads a typed view; storage owns the append and returned
// prefix. No model binding or trajectory container is part of Cell.
struct EventContinuation {
    DeterministicEvaluator evaluator;

    EventCell forward(EventBuffer& storage, const EventCell& prefix) const {
        storage.require_current_tail(prefix);
        const auto input = prefix.view<EventView>();
        const auto next = evaluator(input);
        return storage.append_continuation(prefix, next);
    }
};

void print_events(const char* label, const EventCell& cell) {
    const auto events = cell.view<EventView>().tokens;
    std::cout << label << " identity=" << cell.identity() << " prefix=[";
    for (std::size_t i = 0; i < events.count; ++i) {
        if (i != 0) std::cout << ',';
        std::cout << events.data[i];
    }
    std::cout << "] limit=" << cell.limit() << '\n';
}

} // namespace

int main() {
    EventBuffer storage({2, 5, 5}, "cell-K");
    std::size_t prediction_calls = 0;
    const EventContinuation operation{DeterministicEvaluator{prediction_calls}};

    auto initial = storage.cell();
    print_events("initial", initial);

    // Equal identity and prefix length are not sufficient: the handle must
    // refer to this exact storage owner before evaluating or appending.
    EventBuffer other_storage({2, 5, 5}, "cell-K");
    const auto foreign_prefix = other_storage.cell();
    const auto size_before_rejection = storage.size();
    const auto appends_before_rejection = storage.append_count();
    const auto predictions_before_rejection = prediction_calls;
    try {
        (void)operation.forward(storage, foreign_prefix);
        std::cerr << "expected a same-ID prefix from another buffer to be rejected\n";
        return 1;
    } catch (const std::logic_error& error) {
        if (std::string(error.what()) !=
            "cannot append from a prefix owned by another buffer") {
            std::cerr << "foreign prefix failed for the wrong reason: "
                      << error.what() << '\n';
            return 1;
        }
        std::cout << "foreign buffer rejected: " << error.what() << '\n';
    }
    if (storage.size() != size_before_rejection ||
        storage.append_count() != appends_before_rejection ||
        prediction_calls != predictions_before_rejection) {
        std::cerr << "foreign-prefix rejection changed storage or ran prediction\n";
        return 1;
    }

    auto extended = operation.forward(storage, initial);
    print_events("extended", extended);
    print_events("earlier handle", initial);

    const auto prediction_calls_before_branch = prediction_calls;
    try {
        (void)operation.forward(storage, initial);
        std::cerr << "expected the old prefix to be rejected as non-tail\n";
        return 1;
    } catch (const std::logic_error& error) {
        std::cout << "branch rejected: " << error.what() << '\n';
    }
    if (prediction_calls != prediction_calls_before_branch ||
        storage.size() != extended.limit() || storage.append_count() != 1) {
        std::cerr << "stale-prefix rejection changed storage or ran prediction\n";
        return 1;
    }

    auto second = operation.forward(storage, extended);
    print_events("second extension", second);
}
