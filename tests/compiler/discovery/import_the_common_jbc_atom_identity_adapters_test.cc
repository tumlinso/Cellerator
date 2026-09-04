#include <Cellerator/compiler/discovery/import_the_common_jbc_atom_identity_adapters_v1.hh>

#include <cassert>
#include <cstddef>
#include <cstdint>

using namespace Cellerator::compiler::discovery;

namespace {

template <typename Tag>
struct source_cellshard_strong_id {
    std::uint64_t value = 0;
};

}  // namespace

int main() {
    static_assert(sizeof(source_cellshard_strong_id<struct atom_tag>) ==
                  sizeof(cellshard_strong_id_view_v1));
    static_assert(offsetof(persistent_atom_identity_v1, producer_namespace) == 0);
    static_assert(offsetof(persistent_atom_identity_v1, local_identity) == 8);

    atom_identity_validation_code_v1 status{};
    const auto source = cellshard_strong_id_view_v1{0xfedcba9876543210ULL};
    const auto first = adapt_cellshard_strong_id_v1(41, source, &status);
    assert(status == atom_identity_validation_code_v1::success);
    assert(first.producer_namespace == 41);
    assert(first.local_identity == source.value);

    const auto same = adapt_cellshard_strong_id_v1(41, source);
    const auto distinct_namespace = adapt_cellshard_strong_id_v1(42, source);
    const auto distinct_local = adapt_cellshard_strong_id_v1(41, {source.value - 1});
    assert(first == same);
    assert(first != distinct_namespace);
    assert(first != distinct_local);
    assert(persistent_atom_identity_less_v1(first, distinct_namespace));
    assert(persistent_atom_identity_less_v1(distinct_local, first));

    const auto species = make_cellerator_species_identity_v1(
        atom_species_v1::state_neighborhood);
    assert(species == persistent_atom_identity_v1({1, 8}));
    const atom_identity_contract_v1 contract{
        first, species, atom_state_kind_v1::biological_state};
    assert(validate_atom_identity_contract_v1(contract) ==
           atom_identity_validation_code_v1::success);

    assert(!valid_atom_species_v1(static_cast<atom_species_v1>(0)));
    assert(!valid_atom_state_kind_v1(static_cast<atom_state_kind_v1>(5)));
    assert(!adapt_cellshard_strong_id_v1(0, source, &status).producer_namespace);
    assert(status == atom_identity_validation_code_v1::invalid_producer_namespace);
    assert(!adapt_cellshard_strong_id_v1(41, {}, &status).local_identity);
    assert(status == atom_identity_validation_code_v1::invalid_legacy_identity);
}
