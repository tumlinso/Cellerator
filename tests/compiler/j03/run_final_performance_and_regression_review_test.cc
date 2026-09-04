#include "cmake/compiler/part_one_acceptance_v1.hh"

#include <cassert>

int main() {
    using namespace cellerator::compiler::acceptance::v1;
    assert(performance_review.size() == 6);
    for (const auto& item : performance_review) {
        assert(!item.subject.empty());
        assert(!item.baseline.empty());
        assert(!item.identity.empty());
        assert(item.disposition == "retained" || item.disposition == "accepted" ||
               item.disposition == "not-promoted");
    }
}
