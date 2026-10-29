#pragma once

#include <Cellerator/modeling/entity_layout.hh>
#include <Cellerator/modeling/sparse_relation.hh>

#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>

namespace cellerator::modeling {

enum class JointReduction { sum, mean, product, minimum, maximum };

// A host physical realization of selected logical hyperedges. Topology,
// activity and value order are frozen here; numerical values are supplied at
// invocation and do not trigger a rebuild. Repeated incidences remain repeated.
class IncidencePlan final {
public:
    IncidencePlan(ActiveWorklist worklist, EntityLayout layout)
        : worklist_(std::move(worklist)), layout_(std::move(layout)), offsets_{0} {
        for (const auto ordinal : worklist_.ordinals()) {
            const auto& edge = worklist_.relation()->edges()[ordinal];
            for (std::uint32_t i = 0; i < edge.participant_count; ++i) {
                const auto& participant = worklist_.relation()->participants()[edge.participant_begin + i];
                slots_.push_back(layout_.slot(participant.entity_id));
                roles_.push_back(participant.role);
            }
            offsets_.push_back(slots_.size());
        }
    }
    const ActiveWorklist& worklist() const noexcept { return worklist_; }
    const EntityLayout& layout() const noexcept { return layout_; }
    const std::vector<std::size_t>& offsets() const noexcept { return offsets_; }
    const std::vector<std::size_t>& slots() const noexcept { return slots_; }
    const std::vector<std::uint32_t>& roles() const noexcept { return roles_; }
    std::size_t edge_count() const noexcept { return worklist_.size(); }
    std::size_t incidence_count() const noexcept { return slots_.size(); }

private:
    ActiveWorklist worklist_;
    EntityLayout layout_;
    std::vector<std::size_t> offsets_, slots_;
    std::vector<std::uint32_t> roles_;
};

inline std::size_t checked_value_count(std::size_t rows, std::size_t features) {
    if (features == 0) throw std::invalid_argument("feature count must be positive");
    if (rows > std::numeric_limits<std::size_t>::max() / features)
        throw std::length_error("value shape overflow");
    return rows * features;
}

// Storage, accumulator and output types are separate template parameters.
// The caller chooses numeric precision; no universal half implementation or
// CUDA execution is implied by these host reference algorithms.
template<class Storage>
std::vector<Storage> gather(const IncidencePlan& plan, const std::vector<Storage>& values,
                            std::size_t features = 1) {
    if (values.size() != checked_value_count(plan.layout().size(), features))
        throw std::invalid_argument("entity value shape does not match layout");
    std::vector<Storage> result(checked_value_count(plan.incidence_count(), features));
    for (std::size_t incidence = 0; incidence < plan.incidence_count(); ++incidence)
        std::copy_n(values.begin() + plan.slots()[incidence] * features, features,
                    result.begin() + incidence * features);
    return result;
}

template<class Storage, class Accumulator = Storage, class Output = Accumulator>
std::vector<Output> joint_reduce(const IncidencePlan& plan, const std::vector<Storage>& values,
                                std::size_t features = 1,
                                JointReduction reduction = JointReduction::sum,
                                std::optional<std::uint32_t> role = std::nullopt) {
    if (values.size() != checked_value_count(plan.layout().size(), features))
        throw std::invalid_argument("entity value shape does not match layout");
    if (reduction != JointReduction::sum && reduction != JointReduction::mean
        && reduction != JointReduction::product && reduction != JointReduction::minimum
        && reduction != JointReduction::maximum)
        throw std::invalid_argument("unknown joint reduction");
    std::vector<Output> result(checked_value_count(plan.edge_count(), features));
    // Direct sparse reads avoid materializing the gather buffer.
    for (std::size_t edge = 0; edge < plan.edge_count(); ++edge) {
        std::size_t count = 0;
        for (std::size_t i = plan.offsets()[edge]; i < plan.offsets()[edge + 1]; ++i)
            if (!role || plan.roles()[i] == *role) ++count;
        if (count == 0 && reduction != JointReduction::sum && reduction != JointReduction::product)
            throw std::invalid_argument("mean/minimum/maximum requires a participating incidence per edge");
        for (std::size_t feature = 0; feature < features; ++feature) {
            Accumulator accumulator = reduction == JointReduction::product ? Accumulator{1} : Accumulator{0};
            bool first = true;
            for (std::size_t i = plan.offsets()[edge]; i < plan.offsets()[edge + 1]; ++i) {
                if (role && plan.roles()[i] != *role) continue;
                const auto value = static_cast<Accumulator>(values[plan.slots()[i] * features + feature]);
                switch (reduction) {
                case JointReduction::sum: case JointReduction::mean: accumulator += value; break;
                case JointReduction::product: accumulator *= value; break;
                // Explicit comparisons preserve an encountered NaN rather than
                // silently discarding it through an ordered min/max helper.
                case JointReduction::minimum:
                    if (first || value < accumulator || value != value) accumulator = value;
                    break;
                case JointReduction::maximum:
                    if (first || value > accumulator || value != value) accumulator = value;
                    break;
                }
                first = false;
            }
            if (reduction == JointReduction::mean) accumulator /= static_cast<Accumulator>(count);
            result[edge * features + feature] = static_cast<Output>(accumulator);
        }
    }
    return result;
}

template<class Storage, class Accumulator = Storage, class Output = Accumulator>
std::vector<Output> scatter_add(const IncidencePlan& plan, const std::vector<Storage>& edge_values,
                               std::size_t features = 1,
                               std::optional<std::uint32_t> role = std::nullopt) {
    if (edge_values.size() != checked_value_count(plan.edge_count(), features))
        throw std::invalid_argument("edge value shape does not match worklist");
    std::vector<Accumulator> accumulated(checked_value_count(plan.layout().size(), features), Accumulator{0});
    for (std::size_t edge = 0; edge < plan.edge_count(); ++edge)
        for (std::size_t i = plan.offsets()[edge]; i < plan.offsets()[edge + 1]; ++i) {
            if (role && plan.roles()[i] != *role) continue;
            for (std::size_t feature = 0; feature < features; ++feature)
                accumulated[plan.slots()[i] * features + feature]
                    += static_cast<Accumulator>(edge_values[edge * features + feature]);
        }
    return {accumulated.begin(), accumulated.end()};
}

// Fused read/reduce/scatter without an incidence-sized intermediate. This
// creates one edge-sized temporary and one fresh entity output, never mutating
// the input state. Distinct input and output role filters support directed
// hyperedges and overlapping group aggregation.
template<class Storage, class Accumulator = Storage, class Output = Accumulator>
std::vector<Output> joint_reduce_scatter(
    const IncidencePlan& plan, const std::vector<Storage>& values, std::size_t features = 1,
    JointReduction reduction = JointReduction::sum,
    std::optional<std::uint32_t> input_role = std::nullopt,
    std::optional<std::uint32_t> output_role = std::nullopt) {
    const auto edge_values = joint_reduce<Storage, Accumulator, Accumulator>(
        plan, values, features, reduction, input_role);
    return scatter_add<Accumulator, Accumulator, Output>(plan, edge_values, features, output_role);
}

} // namespace cellerator::modeling
