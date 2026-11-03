#pragma once

#include <Cellerator/containers/value_shape.hh>
#include <Cellerator/execution/dependency/incidence.hh>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <vector>

namespace cellerator::ops::incidence {

enum class JointReduction { sum, mean, product, minimum, maximum };

// Storage, accumulator and output types are separate template parameters.
// The caller chooses numeric precision; these are host reference algorithms,
// not a universal dtype policy or an implicit CUDA implementation.
template<class Storage>
std::vector<Storage> gather(const ::cellerator::execution::PreparedIncidence& plan,
                            const std::vector<Storage>& values,
                            std::size_t features = 1) {
    if (values.size() != ::cellerator::containers::checked_value_count(plan.layout().size(), features))
        throw std::invalid_argument("entity value shape does not match layout");
    std::vector<Storage> result(::cellerator::containers::checked_value_count(plan.incidence_count(), features));
    for (std::size_t incidence = 0; incidence < plan.incidence_count(); ++incidence)
        std::copy_n(values.begin() + plan.slots()[incidence] * features, features,
                    result.begin() + incidence * features);
    return result;
}

template<class Storage, class Accumulator = Storage, class Output = Accumulator>
std::vector<Output> joint_reduce(
    const ::cellerator::execution::PreparedIncidence& plan,
    const std::vector<Storage>& values, std::size_t features = 1,
    JointReduction reduction = JointReduction::sum,
    std::optional<std::uint32_t> role = std::nullopt) {
    if (values.size() != ::cellerator::containers::checked_value_count(plan.layout().size(), features))
        throw std::invalid_argument("entity value shape does not match layout");
    if (reduction != JointReduction::sum && reduction != JointReduction::mean
        && reduction != JointReduction::product && reduction != JointReduction::minimum
        && reduction != JointReduction::maximum)
        throw std::invalid_argument("unknown joint reduction");
    std::vector<Output> result(::cellerator::containers::checked_value_count(plan.edge_count(), features));
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
std::vector<Output> scatter_add(
    const ::cellerator::execution::PreparedIncidence& plan,
    const std::vector<Storage>& edge_values, std::size_t features = 1,
    std::optional<std::uint32_t> role = std::nullopt) {
    if (edge_values.size() != ::cellerator::containers::checked_value_count(plan.edge_count(), features))
        throw std::invalid_argument("edge value shape does not match worklist");
    std::vector<Accumulator> accumulated(
        ::cellerator::containers::checked_value_count(plan.layout().size(), features), Accumulator{0});
    for (std::size_t edge = 0; edge < plan.edge_count(); ++edge)
        for (std::size_t i = plan.offsets()[edge]; i < plan.offsets()[edge + 1]; ++i) {
            if (role && plan.roles()[i] != *role) continue;
            for (std::size_t feature = 0; feature < features; ++feature)
                accumulated[plan.slots()[i] * features + feature]
                    += static_cast<Accumulator>(edge_values[edge * features + feature]);
        }
    return {accumulated.begin(), accumulated.end()};
}

// Preserves the existing two-stage behavior: one edge-sized reduction
// temporary followed by a scatter into a fresh entity output. A fused kernel
// is a separate operation with its own correctness and cost evidence.
template<class Storage, class Accumulator = Storage, class Output = Accumulator>
std::vector<Output> joint_reduce_scatter(
    const ::cellerator::execution::PreparedIncidence& plan,
    const std::vector<Storage>& values, std::size_t features = 1,
    JointReduction reduction = JointReduction::sum,
    std::optional<std::uint32_t> input_role = std::nullopt,
    std::optional<std::uint32_t> output_role = std::nullopt) {
    const auto edge_values = joint_reduce<Storage, Accumulator, Accumulator>(
        plan, values, features, reduction, input_role);
    return scatter_add<Accumulator, Accumulator, Output>(plan, edge_values, features, output_role);
}

} // namespace cellerator::ops::incidence
