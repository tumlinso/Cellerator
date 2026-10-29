#include <Cellerator/execution/conditional/dispatch.hh>

#include <cmath>
#include <map>
#include <numeric>

namespace cellerator::execution::conditional {

DispatchPlan::DispatchPlan(std::vector<Selection> selections) : selections_(std::move(selections)) {
    for (const auto& selection : selections_)
        if (!std::isfinite(selection.weight)) throw std::invalid_argument("selected route weights must be finite");

    dispatch_to_selection_.resize(size());
    std::iota(dispatch_to_selection_.begin(), dispatch_to_selection_.end(), 0);
    std::stable_sort(dispatch_to_selection_.begin(), dispatch_to_selection_.end(),
        [&](auto left, auto right) { return selections_[left].expert < selections_[right].expert; });
    selection_to_dispatch_.resize(size());
    dispatch_origins_.reserve(size());
    expert_offsets_.push_back(0);
    for (std::size_t dispatch = 0; dispatch < size(); ++dispatch) {
        const auto selection = dispatch_to_selection_[dispatch];
        const auto& item = selections_[selection];
        selection_to_dispatch_[selection] = dispatch;
        dispatch_origins_.push_back(item.origin);
        if (experts_.empty() || experts_.back() != item.expert) {
            if (!experts_.empty()) expert_offsets_.push_back(dispatch);
            experts_.push_back(item.expert);
        }
    }
    if (size()) expert_offsets_.push_back(size());

    std::map<Origin, std::size_t> origin_slots;
    selection_to_origin_.reserve(size());
    for (const auto& selection : selections_) {
        const auto inserted = origin_slots.emplace(selection.origin, origins_.size());
        if (inserted.second) origins_.push_back(selection.origin);
        selection_to_origin_.push_back(inserted.first->second);
    }
    dispatch_origin_slots_.reserve(size());
    dispatch_weights_.reserve(size());
    for (const auto selection : dispatch_to_selection_) {
        dispatch_origin_slots_.push_back(selection_to_origin_[selection]);
        dispatch_weights_.push_back(selections_[selection].weight);
    }
    origin_offsets_.assign(origin_count() + 1, 0);
    for (const auto origin : selection_to_origin_) ++origin_offsets_[origin + 1];
    for (std::size_t origin = 0; origin < origin_count(); ++origin)
        origin_offsets_[origin + 1] += origin_offsets_[origin];
    auto cursor = origin_offsets_;
    origin_dispatch_indices_.resize(size());
    // Iteration in original selection order fixes reverse-combine summation
    // order even when one origin was dispatched to many expert/revision groups.
    for (std::size_t selection = 0; selection < size(); ++selection)
        origin_dispatch_indices_[cursor[selection_to_origin_[selection]]++] = selection_to_dispatch_[selection];
}

} // namespace cellerator::execution::conditional
