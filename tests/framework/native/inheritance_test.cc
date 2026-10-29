#ifdef NDEBUG
#undef NDEBUG
#endif

#include <Cellerator/modeling/inheritance.hh>

#include <cassert>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

using cellerator::modeling::inheritance::BinomialPolicy;
using cellerator::modeling::inheritance::FractionPolicy;
using cellerator::modeling::inheritance::MemberInheritance;
using cellerator::modeling::inheritance::MemberReset;
using cellerator::modeling::inheritance::binomial_partition;
using cellerator::modeling::inheritance::fieldwise_inherit;
using cellerator::modeling::inheritance::fraction_partition;
using cellerator::modeling::inheritance::member_custom;

struct State {
    std::uint64_t count = 0;
    double amount = 0.0;
    int accessible = 0;
    std::vector<std::string> memory;
};

int main() {
    std::mt19937_64 random(73);
    const auto [a, b] = binomial_partition<std::uint64_t>(19, 0.37, random);
    assert(a + b == 19);
    const auto [x, y] = fraction_partition(std::vector<double>{8.0, 5.0}, 0.25);
    assert(x.size() == 2 && y.size() == 2);
    assert(x[0] + y[0] == 8.0 && x[1] + y[1] == 5.0);

    const State parent{19, 8.0, 7, {"earlier", "recent"}};
    const auto daughters = fieldwise_inherit(
        parent,
        random,
        MemberInheritance<&State::count, BinomialPolicy>{BinomialPolicy{0.4}},
        MemberInheritance<&State::amount, FractionPolicy>{FractionPolicy{0.25}},
        MemberReset<&State::accessible, int>{1, 0},
        member_custom<&State::memory>(
            [](const auto& memory, auto&) {
                auto first = memory;
                auto second = memory;
                second.push_back("daughter-specific");
                return std::make_pair(std::move(first), std::move(second));
            }));

    assert(daughters.first.count + daughters.second.count == parent.count);
    assert(daughters.first.amount + daughters.second.amount == parent.amount);
    assert(daughters.first.accessible == 1 && daughters.second.accessible == 0);
    assert(daughters.first.memory.size() == 2 && daughters.second.memory.size() == 3);
    assert(parent.memory.size() == 2);
    assert(daughters.first.memory.data() != daughters.second.memory.data());

    auto invalid = false;
    try { (void)binomial_partition(3, 1.1, random); }
    catch (const std::invalid_argument&) { invalid = true; }
    assert(invalid);
    return 0;
}
