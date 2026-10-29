#include "cmake/compiler/part_one_acceptance_v1.hh"

#include <cassert>
#include <set>
#include <string_view>

int main() {
    using namespace cellerator::compiler::acceptance::v1;
    const std::set<std::string_view> bundle(reproducibility_bundle.begin(), reproducibility_bundle.end());
    assert(bundle.size() == reproducibility_bundle.size());
    assert(bundle.count("source-revision") == 1);
    assert(bundle.count("profile-fixture") == 1);
    assert(bundle.count("benchmark-contract") == 1);
    assert(bundle.count("sdk-consumer") == 1);
    assert(bundle.count("provenance") == 1);
}
