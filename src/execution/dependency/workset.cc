#include <Cellerator/execution/dependency/workset.hh>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace cellerator::execution {

ActiveWorklist::ActiveWorklist(
    std::shared_ptr<const geometry::dependency::ExplicitPattern> pattern,
    std::vector<std::size_t> ordinals, std::uint64_t activity_generation)
    : pattern_(std::move(pattern)), ordinals_(std::move(ordinals)), generation_(activity_generation) {
    if (!pattern_) throw std::invalid_argument("worklist pattern cannot be null");
    std::sort(ordinals_.begin(), ordinals_.end());
    if (std::adjacent_find(ordinals_.begin(), ordinals_.end()) != ordinals_.end())
        throw std::invalid_argument("worklist contains duplicate edges");
    if (!ordinals_.empty() && ordinals_.back() >= pattern_->size())
        throw std::out_of_range("worklist ordinal outside explicit pattern");
}

std::vector<std::uint64_t> ActiveWorklist::edge_ids() const {
    std::vector<std::uint64_t> result;
    result.reserve(size());
    for (const auto ordinal : ordinals_)
        result.push_back(pattern_->edges()[ordinal].logical_edge_id);
    return result;
}

ActivityState::ActivityState(
    std::shared_ptr<const geometry::dependency::ExplicitPattern> pattern,
    bool initially_active)
    : pattern_(std::move(pattern)) {
    if (!pattern_) throw std::invalid_argument("activity pattern cannot be null");
    active_.assign(pattern_->size(), initially_active ? 1 : 0);
}

void ActivityState::set_active(const std::vector<std::uint64_t>& ids, bool active) {
    std::vector<std::size_t> ordinals;
    ordinals.reserve(ids.size());
    bool changed = false;
    for (const auto id : ids) {
        const auto ordinal = pattern_->ordinal(id);
        ordinals.push_back(ordinal);
        changed = changed || (active_[ordinal] != static_cast<std::uint8_t>(active));
    }
    if (!changed) return;
    if (generation_ == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("activity generation overflow");
    for (const auto ordinal : ordinals) active_[ordinal] = active;
    ++generation_;
}

ActiveWorklist ActivityState::worklist() const {
    std::vector<std::size_t> ordinals;
    for (std::size_t i = 0; i < active_.size(); ++i)
        if (active_[i]) ordinals.push_back(i);
    return {pattern_, std::move(ordinals), generation_};
}

ActiveWorklist ActivityState::query(
    const std::vector<containers::PostingTerm>& terms, bool require_all) const {
    auto ordinals = pattern_->query(terms, require_all);
    ordinals.erase(std::remove_if(ordinals.begin(), ordinals.end(),
                                 [&](auto ordinal) { return !active_[ordinal]; }), ordinals.end());
    return {pattern_, std::move(ordinals), generation_};
}

ActivityState ActivityState::rebase(
    std::shared_ptr<const geometry::dependency::ExplicitPattern> replacement,
    bool new_edges_active) const {
    if (!replacement || replacement->pattern_id() != pattern_->pattern_id()
        || replacement->structure_id() != pattern_->structure_id()
        || replacement->structure_epoch() <= pattern_->structure_epoch())
        throw std::invalid_argument("activity rebase requires a newer epoch of the same pattern/structure");
    if (generation_ == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("activity generation overflow");
    ActivityState result(std::move(replacement), new_edges_active);
    std::size_t old = 0;
    for (std::size_t next = 0; next < result.pattern_->size(); ++next) {
        const auto id = result.pattern_->edges()[next].logical_edge_id;
        while (old < pattern_->size() && pattern_->edges()[old].logical_edge_id < id) ++old;
        if (old < pattern_->size() && pattern_->edges()[old].logical_edge_id == id)
            result.active_[next] = active_[old];
    }
    result.generation_ = generation_ + 1;
    return result;
}

} // namespace cellerator::execution
