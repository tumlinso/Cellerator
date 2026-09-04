#include "cmake/compiler/part_one_acceptance_v1.hh"

#include <cassert>
#include <string_view>

int main() {
    using namespace cellerator::compiler::acceptance::v1;
    assert(language_conformance.size() == 8);
    assert(language_conformance.front() == "file-local pragma");
    assert(language_conformance[2] == "profile required");
    assert(language_conformance[5] == "ordinary C++ preserved");
    assert(language_conformance.back() == "implementation-defined target costs");
}
