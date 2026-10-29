#pragma once

#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace cellerator::modeling::inheritance {

// Fractions are caller-selected. Daughter two receives the arithmetic
// remainder, preserving the declared total up to floating-point arithmetic.
inline std::pair<double, double> fraction_partition(double value, double fraction) {
    if (!std::isfinite(value)) throw std::invalid_argument("partition value must be finite");
    if (!std::isfinite(fraction) || fraction < 0.0 || fraction > 1.0)
        throw std::invalid_argument("partition fraction must be finite and in [0, 1]");
    const auto first = value * fraction;
    return {first, value - first};
}

// Random is caller-owned and may be a host engine or an adapter around another
// RNG. The second daughter receives the remainder, so every draw conserves n.
template<class Integer, class Random>
std::pair<Integer, Integer> binomial_partition(Integer n, double fraction, Random& random) {
    static_assert(std::is_integral_v<Integer> && !std::is_same_v<Integer, bool>,
                  "binomial partition requires an integer count type");
    if constexpr (std::is_signed_v<Integer>) {
        if (n < 0) throw std::invalid_argument("count must be nonnegative");
    }
    if (!std::isfinite(fraction) || fraction < 0.0 || fraction > 1.0)
        throw std::invalid_argument("partition fraction must be finite and in [0, 1]");
    std::binomial_distribution<Integer> draw(n, fraction);
    const auto first = draw(random);
    return {first, static_cast<Integer>(n - first)};
}

struct BinomialPolicy {
    double fraction = 0.5;

    template<class Integer, class Random>
    std::pair<Integer, Integer> operator()(Integer value, Random& random) const {
        return binomial_partition(value, fraction, random);
    }
};

struct FractionPolicy {
    double fraction = 0.5;

    template<class Number, class Random>
    std::pair<Number, Number> operator()(Number value, Random&) const {
        static_assert(std::is_floating_point_v<Number>,
                      "FractionPolicy requires a floating-point field");
        if (!std::isfinite(value)) throw std::invalid_argument("partition value must be finite");
        if (!std::isfinite(fraction) || fraction < 0.0 || fraction > 1.0)
            throw std::invalid_argument("partition fraction must be finite and in [0, 1]");
        const auto first = static_cast<Number>(value * fraction);
        return {first, static_cast<Number>(value - first)};
    }
};

template<class Integer, class Random>
std::vector<std::pair<Integer, Integer>> binomial_partition(
    const std::vector<Integer>& counts, double fraction, Random& random) {
    std::vector<std::pair<Integer, Integer>> result;
    result.reserve(counts.size());
    for (const auto count : counts)
        result.push_back(binomial_partition(count, fraction, random));
    return result;
}

template<class Number>
std::pair<std::vector<Number>, std::vector<Number>> fraction_partition(
    const std::vector<Number>& values, double fraction) {
    static_assert(std::is_floating_point_v<Number>,
                  "continuous fraction partition requires floating-point values");
    if (!std::isfinite(fraction) || fraction < 0.0 || fraction > 1.0)
        throw std::invalid_argument("partition fraction must be finite and in [0, 1]");
    std::vector<Number> left, right;
    left.reserve(values.size());
    right.reserve(values.size());
    for (const auto value : values) {
        if (!std::isfinite(value)) throw std::invalid_argument("partition values must be finite");
        const auto first = static_cast<Number>(value * fraction);
        left.push_back(first);
        right.push_back(static_cast<Number>(value - first));
    }
    return {std::move(left), std::move(right)};
}

// Rules address fields through member pointers supplied by the caller. Rules
// own interpretation and custom state assignment; this helper supplies paired
// value copies before any rule mutates either daughter.
template<class State, class Random, class... Rules>
std::pair<State, State> fieldwise_inherit(
    const State& parent, Random& random, const Rules&... rules) {
    State first(parent);
    State second(parent);
    (rules.apply(parent, first, second, random), ...);
    return {std::move(first), std::move(second)};
}

template<auto ParentMember, class Policy, auto FirstMember = ParentMember,
         auto SecondMember = ParentMember>
struct MemberInheritance {
    Policy policy;

    template<class State, class Random>
    void apply(const State& parent, State& first, State& second, Random& random) const {
        const auto daughters = policy(parent.*ParentMember, random);
        first.*FirstMember = daughters.first;
        second.*SecondMember = daughters.second;
    }
};

template<auto ParentMember, class Value, auto FirstMember = ParentMember,
         auto SecondMember = ParentMember>
struct MemberReset {
    Value first_value;
    Value second_value;

    template<class State, class Random>
    void apply(const State&, State& first, State& second, Random&) const {
        first.*FirstMember = first_value;
        second.*SecondMember = second_value;
    }
};

template<auto ParentMember, class Function, auto FirstMember = ParentMember,
         auto SecondMember = ParentMember>
struct MemberCustom {
    Function function;

    template<class State, class Random>
    void apply(const State& parent, State& first, State& second, Random& random) const {
        auto daughters = function(parent.*ParentMember, random);
        first.*FirstMember = std::move(daughters.first);
        second.*SecondMember = std::move(daughters.second);
    }
};

template<auto ParentMember, class Function,
         auto FirstMember = ParentMember, auto SecondMember = ParentMember>
auto member_custom(Function function) {
    return MemberCustom<ParentMember, Function, FirstMember, SecondMember>{std::move(function)};
}

// State must have independent value-copy semantics for mutable owned fields.
// Pointer- or handle-owning states supply a custom copy operation/policy.

} // namespace cellerator::modeling::inheritance
