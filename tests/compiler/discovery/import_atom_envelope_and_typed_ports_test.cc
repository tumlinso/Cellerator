#include <Cellerator/compiler/discovery/import_atom_envelope_and_typed_ports_v1.hh>

#include <cassert>

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

planning_atom_envelope_v1 fixture() {
    return {
        {id(1), make_cellerator_species_identity_v1(atom_species_v1::executable),
         atom_state_kind_v1::biological_state},
        atom_certification_state_v1::certified,
        {id(30), 12, true},
        {port(40, atom_port_direction_v1::input),
         port(41, atom_port_direction_v1::output)},
        {{id(50), id(51), 4}, {id(50), id(52), 4}},
        {{id(60), 5, atom_dependency_effect_v1::values},
         {id(61), 6, atom_dependency_effect_v1::correctness}},
        id(70),
        8,
    };
}

}  // namespace

int main() {
    const auto original = fixture();
    assert(validate_atom_envelope_v1(original) == atom_envelope_status_v1::success);
    planning_atom_envelope_v1 round_trip;
    assert(clone_atom_envelope_v1(original, &round_trip) ==
           atom_envelope_status_v1::success);
    assert(round_trip.identities.atom == original.identities.atom);
    assert(round_trip.identities.species == original.identities.species);
    assert(round_trip.exact_coverage.logical_member_count == 12);
    assert(round_trip.ports.size() == 2);
    assert(round_trip.ports[1].direction == atom_port_direction_v1::output);
    assert(round_trip.planes[1].plane_identity == id(52));
    assert(round_trip.dependencies[1].effect ==
           atom_dependency_effect_v1::correctness);
    assert(round_trip.lineage_identity == id(70));

    auto shape_only = original;
    shape_only.ports[1].order_identity = id(999);
    planning_atom_envelope_v1 shape_round_trip;
    assert(clone_atom_envelope_v1(shape_only, &shape_round_trip) ==
           atom_envelope_status_v1::success);
    assert(shape_round_trip.ports[1].order_identity == id(999));
    assert(shape_round_trip.ports[1].order_identity !=
           round_trip.ports[1].order_identity);

    auto candidate = original;
    candidate.certification = atom_certification_state_v1::candidate;
    candidate.exact_coverage.certified_exact = false;
    assert(validate_atom_envelope_v1(candidate) == atom_envelope_status_v1::success);
    candidate.certification = atom_certification_state_v1::certified;
    assert(validate_atom_envelope_v1(candidate) ==
           atom_envelope_status_v1::invalid_coverage);

    auto duplicate_port = original;
    duplicate_port.ports[1].port_identity = duplicate_port.ports[0].port_identity;
    assert(validate_atom_envelope_v1(duplicate_port) ==
           atom_envelope_status_v1::unordered_ports);
}
