#include <Cellerator/compiler/discovery/import_the_common_jbc_atom_identity_adapters_v1.hh>

namespace Cellerator::compiler::discovery {

persistent_atom_identity_v1 make_cellerator_species_identity_v1(
    atom_species_v1 species) noexcept {
    if (!valid_atom_species_v1(species)) return {};
    return {cellerator_atom_provider_namespace_v1,
            static_cast<std::uint64_t>(species)};
}

persistent_atom_identity_v1 adapt_cellshard_strong_id_v1(
    std::uint64_t producer_namespace,
    cellshard_strong_id_view_v1 legacy_identity,
    atom_identity_validation_code_v1* status) noexcept {
    if (producer_namespace == 0) {
        if (status != nullptr)
            *status = atom_identity_validation_code_v1::invalid_producer_namespace;
        return {};
    }
    if (legacy_identity.value == 0) {
        if (status != nullptr)
            *status = atom_identity_validation_code_v1::invalid_legacy_identity;
        return {};
    }
    if (status != nullptr) *status = atom_identity_validation_code_v1::success;
    return {producer_namespace, legacy_identity.value};
}

atom_identity_validation_code_v1 validate_atom_identity_contract_v1(
    const atom_identity_contract_v1& contract) noexcept {
    if (!valid_persistent_atom_identity_v1(contract.atom))
        return atom_identity_validation_code_v1::invalid_atom_identity;
    if (!valid_persistent_atom_identity_v1(contract.species))
        return atom_identity_validation_code_v1::invalid_species_identity;
    if (!valid_atom_state_kind_v1(contract.state))
        return atom_identity_validation_code_v1::invalid_state;
    return atom_identity_validation_code_v1::success;
}

}  // namespace Cellerator::compiler::discovery
