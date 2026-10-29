#pragma once

#include <Cellerator/modeling/incidence_plan.hh>

namespace cellerator::modeling {

// Optional typed, identity-addressed host field. The structure/order is fixed;
// sparse value writes advance only the value generation. Callers that already
// own arrays can invoke the numerical functions without using this container.
template<class Storage>
class ValueField final {
public:
    ValueField(EntityLayout layout, std::size_t features, std::vector<Storage> values)
        : layout_(std::move(layout)), features_(features), values_(std::move(values)) {
        if (values_.size() != checked_value_count(layout_.size(), features_))
            throw std::invalid_argument("field value shape does not match entity layout");
    }
    const EntityLayout& layout() const noexcept { return layout_; }
    std::size_t features() const noexcept { return features_; }
    const std::vector<Storage>& values() const noexcept { return values_; }
    execution::value_generation generation() const noexcept { return {generation_}; }
    void replace(std::vector<Storage> values) {
        if (values.size() != values_.size()) throw std::invalid_argument("replacement field shape mismatch");
        advance_generation();
        values_.swap(values);
    }
    // Duplicate entity writes are rejected to avoid an implicit conflict policy.
    // All identity and shape checks finish before any state is changed.
    void update(const std::vector<std::uint64_t>& ids, const std::vector<Storage>& values) {
        if (values.size() != checked_value_count(ids.size(), features_))
            throw std::invalid_argument("sparse field write shape mismatch");
        std::vector<std::size_t> slots;
        slots.reserve(ids.size());
        for (auto id : ids) slots.push_back(layout_.slot(id));
        auto sorted = slots;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
            throw std::invalid_argument("duplicate entity in sparse field write");
        if (slots.empty()) return;
        advance_generation();
        for (std::size_t i = 0; i < slots.size(); ++i)
            std::copy_n(values.begin() + i * features_, features_, values_.begin() + slots[i] * features_);
    }

private:
    void advance_generation() {
        if (generation_ == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("value generation overflow");
        ++generation_;
    }
    EntityLayout layout_;
    std::size_t features_;
    std::vector<Storage> values_;
    std::uint64_t generation_ = 0;
};

} // namespace cellerator::modeling
