#pragma once

#include <Cellerator/compiler/composition/import_explicit_grammar_compilation_v1.hh>
#include <Cellerator/compiler/composition/import_induced_grammar_as_experimental_search_v1.hh>
#include <Cellerator/compiler/composition/import_multi_parent_derivation_dags_v1.hh>
#include <Cellerator/compiler/composition/import_superatom_promotion_v1.hh>
#include <Cellerator/compiler/composition/import_typed_composition_production_contracts_v1.hh>

#include <cstdint>

namespace Cellerator::compiler::composition {

inline constexpr std::uint32_t grammar_contract_version_v1 = 1;

} // namespace Cellerator::compiler::composition
