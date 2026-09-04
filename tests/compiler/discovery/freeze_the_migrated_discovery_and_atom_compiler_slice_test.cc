#include <Cellerator/compiler/discovery/freeze_the_migrated_discovery_and_atom_compiler_slice_v1.hh>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace Cellerator::compiler::discovery;

namespace {

persistent_atom_identity_v1 id(std::uint64_t value) {
    return {1, value};
}

atom_typed_port_v1 port(std::uint64_t value,
                        atom_port_direction_v1 direction) {
    return {id(value), id(20), id(21), id(22), id(23), id(24), id(25), id(26),
            3, direction};
}

void validate_vertical_slice() {
    cellerator::compiler::profile::v1::profile_compile_state_v1 profile;
    profile.state = {100, 1};
    profile.structure.relation = {200, 1};
    profile.structure.structure_epoch = 7;
    profile.structure.support_count = 6;
    assert(validate_profile_for_discovery_v1(profile) ==
           profile_discovery_status_v1::ready);

    const std::uint64_t offsets[]{0, 3, 6};
    const std::uint64_t sources[]{1, 3, 7, 1, 3, 7};
    const support_relation_view_v1 relation{
        profile_identity_to_atom_v1(profile.structure.relation),
        offsets, sources, 2, 6};
    support_signature_discovery_v1 discovery;
    assert(discover_support_signatures_v1(
               relation, {8, 1, 99, id(12)}, &discovery) ==
           support_signature_status_v1::success);
    assert(discovery.proposals.size() == 1);
    assert(discovery.proposals[0].matching_minima == 8);

    const std::vector<canonical_relation_edge_v1> canonical{
        {id(100), id(200), id(300)},
        {id(101), id(201), id(301)},
    };
    exact_proposal_certificate_v1 certificate;
    assert(certify_proposal_logical_coverage_v1(
               relation.relation_identity, profile.structure.structure_epoch,
               canonical, {{id(400), {id(100), id(101)}}}, 8,
               &certificate) == exact_rescan_status_v1::success);
    assert(certificate.exact_cover);
    assert(!authorizes_execution(certificate));

    certified_atom_request_v1 request{
        {id(500), make_cellerator_species_identity_v1(
                      atom_species_v1::support_signature),
         atom_state_kind_v1::biological_state},
        id(600),
        {port(700, atom_port_direction_v1::input),
         port(701, atom_port_direction_v1::output)},
        {{id(800), id(801), 3}},
        {},
        id(900),
        7,
    };
    planning_atom_envelope_v1 atom;
    assert(build_certified_atom_v1(certificate, request, &atom) ==
           certified_atom_status_v1::success);
    assert(atom.certification == atom_certification_state_v1::certified);
    assert(atom.exact_coverage.logical_member_count == canonical.size());

    auto incomplete = certificate;
    incomplete.exact_cover = false;
    assert(build_certified_atom_v1(incomplete, request, &atom) ==
           certified_atom_status_v1::invalid_certificate);
}

}  // namespace

int main() {
    assert(discovery_contract_version_v1 == 1);
    assert(atom_compiler_contract_version_v1 == 1);
    assert(valid_discovery_atom_slice_receipt_v1());
    const auto& receipt = get_discovery_atom_slice_receipt_v1();
    assert(receipt.migrated_source_record_count == 13);
    assert(receipt.migrated_fixture_source_file_count == 131);
    assert(receipt.provider_family_count == 7);
    assert(receipt.compatibility_retirement_ready);

    validate_vertical_slice();

    constexpr std::uint64_t repeats = 2000;
    const auto begin = std::chrono::steady_clock::now();
    for (std::uint64_t iteration = 0; iteration < repeats; ++iteration) {
        validate_vertical_slice();
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - begin).count();
    std::cout << "profile_to_certified_atoms_v1 repeats=" << repeats
              << " elapsed_ns=" << elapsed
              << " ns_per_iteration=" << (elapsed / repeats)
              << " disposition=validation_only\n";
}
