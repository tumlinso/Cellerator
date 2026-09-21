#pragma once
#include <Cellerator/compute/operation/relation_semantics.hh>
#include <span>
#include <vector>

namespace cellerator::compute::native_numeric {
namespace relation = cellerator::compute::relation;

// Logical edge endpoints in the descriptor's declared source/destination orders.
// prepare copies them; values remain indexed by the original logical edge order.
struct host_topology {
    std::span<const std::uint64_t> sources;
    std::span<const std::uint64_t> destinations;
};
struct value_identity {
    execution::structure_id structure{};
    execution::structure_epoch epoch{};
    execution::order_id edge_order{};
    execution::value_generation generation{};
};

// Synchronous, caller-serialized host owner. Allocation occurs only at prepare.
// No runtime/device/stream ownership. Output overwrite; aliases rejected.
// Value generations label this launch, not a global cache or monotonic publisher.
class host_relation {
public:
    host_relation() = default;
    host_relation(const host_relation&) = default;
    host_relation& operator=(const host_relation&) = default;
    host_relation(host_relation&&) noexcept;
    host_relation& operator=(host_relation&&) noexcept;
    // Strong guarantee: on failure the previous prepared object remains usable.
    relation::status prepare(const relation::operation_descriptor&, host_topology) noexcept;
    relation::status run(value_identity, std::span<const float> values,
                         const relation::axis_descriptor& input_axis, std::span<const float> input,
                         const relation::axis_descriptor& output_axis, std::span<float> output) const noexcept;
    relation::status run(value_identity, std::span<const double> values,
                         const relation::axis_descriptor& input_axis, std::span<const double> input,
                         const relation::axis_descriptor& output_axis, std::span<double> output) const noexcept;
    bool prepared() const noexcept { return prepared_; }
    const relation::operation_descriptor& descriptor() const noexcept { return descriptor_; }
private:
    template<class T> relation::status execute(value_identity, std::span<const T>,
        const relation::axis_descriptor&, std::span<const T>,
        const relation::axis_descriptor&, std::span<T>) const noexcept;
    bool prepared_ = false;
    relation::operation_descriptor descriptor_{};
    std::vector<std::uint64_t> offsets_, edges_, sources_;
};
} // namespace cellerator::compute::native_numeric
