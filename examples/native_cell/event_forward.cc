#include <Cellerator/memory/view.hh>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

namespace {

using Token = uint32_t;
namespace memory = cellerator::memory;

// The caller supplies stable writable storage for the lifetime of its views.
struct EventStorage {
    memory::array_view<Token> events;
    size_t capacity;
    char identity;
};

enum class AppendStatus { appended, foreign_prefix, stale_prefix, full };
struct AppendResult {
    AppendStatus status;
    Token token;
};

AppendResult continue_event(memory::const_array_view<Token>& prefix,
                            const EventStorage& source, EventStorage& storage,
                            size_t& prediction_calls) {
    if (&source != &storage) return {AppendStatus::foreign_prefix, 0};
    if (prefix.data != storage.events.data || prefix.count != storage.events.count)
        return {AppendStatus::stale_prefix, 0};
    if (storage.events.count >= storage.capacity) return {AppendStatus::full, 0};

    // Deterministic example, with one evaluation for each committed event.
    const Token next = prefix.count ? (prefix.data[prefix.count - 1] * 3u + 1u) % 17u : 1u;
    ++prediction_calls;
    storage.events.data[storage.events.count++] = next;
    prefix.count = storage.events.count;
    return {AppendStatus::appended, next};
}

void print_events(const char *label, memory::const_array_view<Token> events, char identity) {
    printf("%s identity=cell-%c prefix=[", label, identity);
    for (size_t i = 0; i < events.count; ++i)
        printf("%s%u", i ? "," : "", unsigned(events.data[i]));
    printf("] limit=%zu\n", events.count);
}

} // namespace

int main() {
    Token events[5] = {2, 5, 5};
    Token other_events[] = {2, 5, 5};
    EventStorage storage{{events, 3, {memory::domain::host}}, 5, 'K'};
    const EventStorage other{{other_events, 3, {memory::domain::host}}, 3, 'K'};
    auto prefix = memory::as_const(storage.events);
    auto foreign = memory::as_const(other.events);
    size_t prediction_calls = 0;
    print_events("initial", prefix, storage.identity);

    // Equal biological identity and length still refer to different buffers.
    const auto rejected = continue_event(foreign, other, storage, prediction_calls);
    if (rejected.status != AppendStatus::foreign_prefix || prediction_calls != 0 ||
        storage.events.count != 3 || foreign.count != 3) {
        fputs("foreign-prefix rejection changed storage or ran prediction\n", stderr);
        return 1;
    }
    puts("foreign buffer rejected: cannot append from a prefix owned by another buffer");

    auto old_prefix = prefix;
    const auto first = continue_event(prefix, storage, storage, prediction_calls);
    if (first.status != AppendStatus::appended || first.token != 16 ||
        prefix.count != 4 || old_prefix.count != 3) {
        fputs("append did not update only the selected prefix boundary\n", stderr);
        return 1;
    }
    printf("appended token=%u\n", unsigned(first.token));
    print_events("extended prefix", prefix, storage.identity);
    print_events("copied earlier prefix", old_prefix, storage.identity);

    const auto stale = continue_event(old_prefix, storage, storage, prediction_calls);
    if (stale.status != AppendStatus::stale_prefix || prediction_calls != 1 ||
        storage.events.count != 4 || old_prefix.count != 3) {
        fputs("stale-prefix rejection changed storage or ran prediction\n", stderr);
        return 1;
    }
    puts("stale prefix rejected: cannot append from a prefix that already has a continuation");

    const auto second = continue_event(prefix, storage, storage, prediction_calls);
    if (second.status != AppendStatus::appended || second.token != 15 ||
        prefix.count != 5 || old_prefix.count != 3) {
        fputs("second append did not extend the current prefix\n", stderr);
        return 1;
    }
    printf("second appended token=%u\n", unsigned(second.token));
    print_events("second extension", prefix, storage.identity);

    const auto full = continue_event(prefix, storage, storage, prediction_calls);
    if (full.status != AppendStatus::full || prediction_calls != 2 ||
        storage.events.count != 5 || prefix.count != 5) {
        fputs("capacity rejection changed storage or ran prediction\n", stderr);
        return 1;
    }
    return 0;
}
