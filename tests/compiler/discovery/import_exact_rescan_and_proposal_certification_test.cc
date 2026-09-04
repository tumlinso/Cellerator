#include <Cellerator/compiler/discovery/import_exact_rescan_and_proposal_certification_v1.hh>

#include <cassert>
#include <vector>

using namespace Cellerator::compiler::discovery;

namespace {

persistent_atom_identity_v1 id(std::uint64_t value) {
    return {1, value};
}

}  // namespace

int main() {
    const std::vector<canonical_relation_edge_v1> canonical{
        {id(10), id(20), id(30)},
        {id(11), id(20), id(31)},
        {id(12), id(21), id(31)},
    };
    exact_proposal_certificate_v1 certificate;
    assert(certify_proposal_logical_coverage_v1(
               id(1), 7, canonical,
               {{id(100), {id(10), id(11)}}, {id(101), {id(12)}}},
               16, &certificate) == exact_rescan_status_v1::success);
    assert(certificate.exact_cover);
    assert(certificate.canonical_owners.size() == 3);
    assert(certificate.omitted_edge_identities.empty());
    assert(certificate.duplicate_receipts.empty());
    assert(!authorizes_execution(certificate));

    assert(certify_proposal_logical_coverage_v1(
               id(1), 7, canonical,
               {{id(100), {id(10), id(11)}}, {id(101), {id(11)}}},
               16, &certificate) == exact_rescan_status_v1::success);
    assert(!certificate.exact_cover);
    assert(certificate.omitted_edge_identities ==
           std::vector<persistent_atom_identity_v1>{id(12)});
    assert(certificate.duplicate_receipts.size() == 1);
    assert(certificate.duplicate_receipts[0].first_proposal_identity == id(100));
    assert(certificate.duplicate_receipts[0].duplicate_proposal_identity == id(101));

    assert(certify_proposal_logical_coverage_v1(
               id(1), 7, canonical, {{id(100), {id(10)}}}, 3, &certificate) ==
           exact_rescan_status_v1::work_bound_exceeded);
    assert(certify_proposal_logical_coverage_v1(
               id(1), 7, canonical, {{id(100), {id(99)}}}, 16, &certificate) ==
           exact_rescan_status_v1::unknown_edge);
}
