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
using EventPrefix = cellerator::cells::EventPrefix<Token, std::string>;
using EventView = cellerator::cells::EventView<Token>;

// This owner controls vector growth and publishes the current descriptor.
// Prefix handles borrow that descriptor and retain their own logical boundary.
class EventBuffer {
public:
    struct AppendResult {
        Token token;
        std::size_t new_limit;
    };

    explicit EventBuffer(std::vector<Token> initial, std::string identity)
        : tokens_(std::move(initial)), identity_(std::move(identity)) {
        refresh_descriptor();
    }

    EventBuffer(const EventBuffer&) = delete;
    EventBuffer& operator=(const EventBuffer&) = delete;
    EventBuffer(EventBuffer&&) = delete;
    EventBuffer& operator=(EventBuffer&&) = delete;

    EventPrefix prefix() const {
        return EventPrefix(descriptor_, 0, tokens_.size(), identity_);
    }

    void require_current_tail(const EventPrefix& prefix) const {
        if (&prefix.buffer_descriptor() != &descriptor_) {
            throw std::logic_error("cannot append from a prefix owned by another buffer");
        }
        if (prefix.identity() != identity_ || prefix.offset() != 0 ||
            prefix.limit() != tokens_.size()) {
            throw std::logic_error(
                "cannot append from a prefix that already has a continuation");
        }
    }

    AppendResult append(const EventPrefix& current, Token token) {
        // Validate again at the storage boundary before changing the buffer.
        require_current_tail(current);
        tokens_.push_back(token);
        refresh_descriptor();
        ++append_count_;
        return {token, tokens_.size()};
    }

    const EventPrefix::buffer_type& descriptor() const & noexcept {
        return descriptor_;
    }
    const EventPrefix::buffer_type& descriptor() const && = delete;
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

// This deterministic callable illustrates the interface only. It is not a
// trained model or a scientific Karyoplast predictor.
struct DeterministicEvaluator {
    std::size_t& calls;

    Token operator()(EventView input) const {
        ++calls;
        if (input.tokens.count == 0) return 1;
        const Token last = input.tokens.data[input.tokens.count - 1];
        return (last * 3u + 1u) % 17u;
    }
};

// The Cell's EventPrefix instance is the mutable logical boundary. Storage
// owns the append; the operation updates this same instance after its append
// succeeds, while copied Cells keep their earlier boundary.
struct EventContinuation {
    DeterministicEvaluator evaluator;

    Token operator()(EventPrefix& prefix, EventBuffer& storage) const {
        storage.require_current_tail(prefix);
        const auto input = prefix.read(cellerator::ViewTag<EventView>{});
        const auto next = evaluator(input);
        const auto result = storage.append(prefix, next);
        prefix.rebind(storage.descriptor(), prefix.offset(), result.new_limit);
        return result.token;
    }
};

void print_events(const char* label, const EventPrefix& prefix) {
    const auto events = prefix.read(cellerator::ViewTag<EventView>{}).tokens;
    const auto identity = prefix.read(
        cellerator::ViewTag<cellerator::CellIdentityView<std::string>>{});
    std::cout << label << " identity=" << identity.value << " prefix=[";
    for (std::size_t i = 0; i < events.count; ++i) {
        if (i != 0) std::cout << ',';
        std::cout << events.data[i];
    }
    std::cout << "] limit=" << prefix.limit() << '\n';
}

} // namespace

int main() {
    EventBuffer storage({2, 5, 5}, "cell-K");
    EventBuffer other_storage({2, 5, 5}, "cell-K");
    std::size_t prediction_calls = 0;
    EventContinuation operation{DeterministicEvaluator{prediction_calls}};

    auto cell = cellerator::make_cell(operation, storage.prefix());
    print_events("initial", cell.instance());

    // Equal identity and prefix length are not sufficient: a foreign handle
    // must be rejected before model evaluation or storage mutation.
    auto foreign_cell = cellerator::make_cell(operation, other_storage.prefix());
    const auto size_before_rejection = storage.size();
    const auto appends_before_rejection = storage.append_count();
    const auto predictions_before_rejection = prediction_calls;
    try {
        (void)foreign_cell.forward(storage);
        std::cerr << "expected a prefix from another buffer to be rejected\n";
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

    // This copy is an old boundary. The successful call extends only `cell`'s
    // instance; the storage owner keeps the same vector and publishes growth.
    auto old_prefix = cell;
    const Token appended = cell.forward(storage);
    if (cell.instance().limit() != 4 || old_prefix.instance().limit() != 3) {
        std::cerr << "append did not update only the selected Cell boundary\n";
        return 1;
    }
    std::cout << "appended token=" << appended << '\n';
    print_events("extended cell", cell.instance());
    print_events("copied earlier prefix", old_prefix.instance());

    const auto predictions_before_stale = prediction_calls;
    try {
        (void)old_prefix.forward(storage);
        std::cerr << "expected copied old prefix to be rejected as stale\n";
        return 1;
    } catch (const std::logic_error& error) {
        std::cout << "stale prefix rejected: " << error.what() << '\n';
    }
    if (prediction_calls != predictions_before_stale ||
        storage.size() != cell.instance().limit() || storage.append_count() != 1) {
        std::cerr << "stale-prefix rejection changed storage or ran prediction\n";
        return 1;
    }

    const Token second = cell.forward(storage);
    std::cout << "second appended token=" << second << '\n';
    print_events("second extension", cell.instance());
}
