#include "cmake/compiler/part_one_acceptance_v1.hh"

#include <cassert>
#include <set>
#include <string_view>

int main() {
    using namespace cellerator::compiler::acceptance::v1;
    const std::set<std::string_view> capabilities(final_capabilities.begin(), final_capabilities.end());
    assert(capabilities.size() == final_capabilities.size());
    assert(capabilities.count("driver") == 1);
    assert(capabilities.count("semantic-ir") == 1);
    assert(capabilities.count("nvidia-object") == 1);
    assert(capabilities.count("jbc-migration") == 1);
    assert(capabilities.count("celleratord") == 1);
    assert(capabilities.count("provenance") == 1);
}
