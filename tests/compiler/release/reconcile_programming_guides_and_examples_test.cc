#include "cmake/compiler/part_one_acceptance_v1.hh"

#include <cassert>
#include <set>
#include <string_view>

int main() {
    using namespace cellerator::compiler::acceptance::v1;
    const std::set<std::string_view> covered(guide_examples.begin(), guide_examples.end());
    assert(covered.size() == guide_examples.size());
    for (const auto required : {"minimal", "profiles", "planning", "realization",
                                "custom-pass", "unsafe-native", "lto", "sdk",
                                "celleratord", "ordinary-cxx"}) {
        assert(covered.count(required) == 1);
    }
}
