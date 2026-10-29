#pragma once

#include <Cellerator/compiler/composition/explicit_grammar_compilation_v1.hh>
#include <Cellerator/compiler/composition/experimental_induced_grammar_search_v1.hh>
#include <Cellerator/compiler/composition/multi_parent_derivation_dags_v1.hh>
#include <Cellerator/compiler/composition/superatom_promotion_v1.hh>
#include <Cellerator/compiler/composition/typed_composition_production_contracts_v1.hh>

#include <cstdint>

namespace Cellerator::compiler::composition {

inline constexpr std::uint32_t grammar_contract_version_v1 = 1;

} // namespace Cellerator::compiler::composition
