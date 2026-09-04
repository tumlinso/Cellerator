#include "cmake/compiler/part_one_acceptance_v1.hh"

#include <cassert>
#include <set>
#include <string_view>

int main() {
    using namespace cellerator::compiler::acceptance::v1;
    const std::set<std::string_view> unique(ir_conformance.begin(), ir_conformance.end());
    assert(unique.size() == ir_conformance.size());
    assert(unique.count("semantic-ir") == 1);
    assert(unique.count("planning-ir") == 1);
    assert(unique.count("realization-ir") == 1);
    assert(unique.count("native-boundary") == 1);
    assert(unique.count("lto") == 1);
}
