#pragma once

#include <Cellerator/structure/identity.hh>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cellerator::modeling {

// Immutable mapping from logical entity IDs to one physical value order.
// Identity is independent of the chosen physical order; permutation_to builds
// an explicit bridge between two layouts over the same domain and entity set.
class EntityLayout final {
public:
    explicit EntityLayout(
        std::vector<std::uint64_t> physical_entity_ids,
        execution::domain_id domain = {1u, 0u},
        execution::order_id order = {1u, 0u})
        : entity_ids_(std::move(physical_entity_ids)), domain_(domain), order_(order) {
        if (domain_.low == 0u && domain_.high == 0u)
            throw std::invalid_argument("entity layout domain identity must be nonzero");
        if (order_.low == 0u && order_.high == 0u)
            throw std::invalid_argument("entity layout order identity must be nonzero");
        slots_.reserve(entity_ids_.size());
        for (std::size_t i = 0; i < entity_ids_.size(); ++i) {
            if (!slots_.emplace(entity_ids_[i], i).second)
                throw std::invalid_argument("entity layout contains a duplicate entity ID");
        }
    }

    std::size_t size() const noexcept { return entity_ids_.size(); }
    const std::vector<std::uint64_t> &entity_ids() const noexcept { return entity_ids_; }

    std::size_t slot(std::uint64_t entity_id) const {
        const auto found = slots_.find(entity_id);
        if (found == slots_.end())
            throw std::out_of_range("entity ID is absent from this layout");
        return found->second;
    }

    execution::domain_id domain() const noexcept { return domain_; }
    execution::order_id order() const noexcept { return order_; }

    // Returned indices are in target physical order: result[i] identifies the
    // source slot whose values belong in target slot i.
    std::vector<std::size_t> permutation_to(const EntityLayout &target) const {
        if (domain_.low != target.domain_.low || domain_.high != target.domain_.high)
            throw std::invalid_argument("entity layouts belong to different domains");
        if (entity_ids_.size() != target.entity_ids_.size())
            throw std::invalid_argument("entity layouts contain different entity sets");
        std::vector<std::size_t> permutation;
        permutation.reserve(target.entity_ids_.size());
        for (const auto id : target.entity_ids_) {
            const auto found = slots_.find(id);
            if (found == slots_.end())
                throw std::invalid_argument("entity layouts contain different entity sets");
            permutation.push_back(found->second);
        }
        return permutation;
    }

    template<typename T>
    std::vector<T> reorder(const std::vector<T> &values,
                           std::size_t features,
                           const EntityLayout &target) const {
        if (features == 0u)
            throw std::invalid_argument("feature count must be positive");
        if (entity_ids_.size() > std::numeric_limits<std::size_t>::max() / features)
            throw std::length_error("entity value shape overflows size_t");
        if (values.size() != entity_ids_.size() * features)
            throw std::invalid_argument("value count does not match entity layout and features");
        const auto permutation = permutation_to(target);
        std::vector<T> result(values.size());
        for (std::size_t target_slot = 0; target_slot < permutation.size(); ++target_slot) {
            const auto source_begin = permutation[target_slot] * features;
            const auto target_begin = target_slot * features;
            for (std::size_t feature = 0; feature < features; ++feature)
                result[target_begin + feature] = values[source_begin + feature];
        }
        return result;
    }

private:
    std::vector<std::uint64_t> entity_ids_;
    execution::domain_id domain_;
    execution::order_id order_;
    std::unordered_map<std::uint64_t, std::size_t> slots_;
};

} // namespace cellerator::modeling
