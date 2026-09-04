#include "cmake/compiler/part_one_acceptance_v1.hh"

#include <cassert>
#include <set>
#include <string_view>

int main() {
    using namespace cellerator::compiler::acceptance::v1;
    const std::set<std::string_view> records(architecture_records.begin(), architecture_records.end());
    assert(records.size() == architecture_records.size());
    assert(records.count("jbc-provenance") == 1);
    assert(records.count("ownership") == 1);
    assert(records.count("part-two-seam") == 1);
}
