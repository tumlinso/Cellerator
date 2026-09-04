#include "cmake/compiler/part_one_acceptance_v1.hh"

#include <cassert>
#include <set>
#include <string_view>

int main() {
    using namespace cellerator::compiler::acceptance::v1;
    const std::set<std::string_view> covered(nvidia_acceptance.begin(), nvidia_acceptance.end());
    assert(covered.size() == nvidia_acceptance.size());
    assert(nvidia_sm == 70);
    assert(nvidia_complete_cost_components == 6);
    assert(covered.count("profile-relation") == 1);
    assert(covered.count("direct-ptx-experiment") == 1);
    assert(covered.count("mixed-lto") == 1);
    assert(covered.count("exact-output") == 1);
}
