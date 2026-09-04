#include <Cellerator/compiler/tooling/implement_document_scheduling_and_cancellation_v1.hh>

#include <algorithm>

namespace Cellerator::compiler::tooling {

document_scheduler_v1::document_scheduler_v1(std::uint64_t debounce_ms,
                                             std::size_t maximum_background)
    : debounce_ms_(debounce_ms), maximum_background_(maximum_background) {}

void document_scheduler_v1::edited(std::string uri, std::uint64_t generation,
                                   std::uint64_t now_ms, bool active) {
    generations_[uri] = generation;
    basic_.erase(std::remove_if(basic_.begin(), basic_.end(), [&](const auto &work) {
        return work.uri == uri;
    }), basic_.end());
    basic_.push_back({std::move(uri), generation, now_ms + debounce_ms_,
                      document_work_kind_v1::parse, active});
    while (basic_.size() > maximum_background_ + 1) {
        const auto oldest_background = std::find_if(basic_.begin(), basic_.end(),
                                                     [](const auto &work) { return !work.active; });
        if (oldest_background == basic_.end()) break;
        basic_.erase(oldest_background);
    }
}

void document_scheduler_v1::request_slow(std::string uri, std::uint64_t generation,
                                         document_work_kind_v1 kind) {
    if (kind != document_work_kind_v1::parse)
        slow_.push_back({std::move(uri), generation, 0, kind, false});
}

std::optional<document_work_v1> document_scheduler_v1::next_basic(std::uint64_t now_ms) {
    const auto selected = std::min_element(basic_.begin(), basic_.end(), [](const auto &a, const auto &b) {
        if (a.active != b.active) return a.active > b.active;
        return a.ready_at_ms < b.ready_at_ms;
    });
    if (selected == basic_.end() || selected->ready_at_ms > now_ms) return std::nullopt;
    auto result = *selected;
    basic_.erase(selected);
    return result;
}

std::optional<document_work_v1> document_scheduler_v1::next_slow() {
    if (slow_.empty()) return std::nullopt;
    auto result = slow_.front();
    slow_.erase(slow_.begin());
    return result;
}

bool document_scheduler_v1::cancelled(const document_work_v1 &work) const noexcept {
    const auto current = generations_.find(work.uri);
    return current == generations_.end() || current->second != work.generation;
}

} // namespace Cellerator::compiler::tooling
