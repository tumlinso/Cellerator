#include "atom_fragment_verifier_v1.hh"

#include <cassert>

namespace verify = cellerator::jbc::verification;

int main() {
    const std::uint64_t left[] = {2u, 0u};
    const std::uint64_t right[] = {3u, 1u};
    const verify::atom_fragment_record_v1 fragments[] = {
        {{1u, 1u}, {10u, 1u}, 0u, 2u, left, 2u},
        {{2u, 1u}, {10u, 1u}, 2u, 2u, right, 2u},
    };
    assert(verify::verify_atom_fragments_v1(
        4u, {10u, 1u}, fragments, 2u));

    auto overlap = fragments[1];
    overlap.logical_begin = 1u;
    const verify::atom_fragment_record_v1 overlapping[] = {
        fragments[0], overlap};
    assert(verify::verify_atom_fragments_v1(
               4u, {10u, 1u}, overlapping, 2u).code ==
        verify::verification_code_v1::overlap);

    const std::uint64_t duplicate[] = {2u, 1u};
    auto duplicate_fragment = fragments[1];
    duplicate_fragment.local_to_global = duplicate;
    const verify::atom_fragment_record_v1 duplicate_recovery[] = {
        fragments[0], duplicate_fragment};
    assert(verify::verify_atom_fragments_v1(
               4u, {10u, 1u}, duplicate_recovery, 2u).code ==
        verify::verification_code_v1::duplicate_recovery);
}
