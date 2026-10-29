#include "cmake/compiler/part_one_acceptance_v1.hh"

#include <cassert>

int main() {
    using namespace cellerator::compiler::acceptance::v1;
    assert(deferred_part_two.size() == 2);
    assert(deferred_part_two[0].name == "general-jit");
    assert(deferred_part_two[1].name == "deep-cellshard-runtime");
    for (const auto& seam : deferred_part_two) {
        assert(!seam.retained_interface.empty());
        assert(!seam.part_one_prerequisite);
    }
}
