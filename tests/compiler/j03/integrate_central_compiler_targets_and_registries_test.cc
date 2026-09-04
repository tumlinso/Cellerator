#include "cmake/compiler/part_one_acceptance_v1.hh"

#include <cassert>
#include <set>
#include <string_view>

int main() {
    using namespace cellerator::compiler::acceptance::v1;
    const std::set<std::string_view> unique(component_registry.begin(), component_registry.end());
    assert(unique.size() == component_registry.size());
    assert(unique.count("semantic-ir") == 1);
    assert(unique.count("planning-ir") == 1);
    assert(unique.count("realization-ir") == 1);
    assert(unique.count("sdk") == 1);
}
