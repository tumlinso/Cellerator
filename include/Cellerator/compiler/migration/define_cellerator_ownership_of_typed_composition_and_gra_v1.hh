#pragma once
#include <array>
#include <cstdint>
#include <string_view>
namespace Cellerator::compiler::migration {
enum class planning_extension_kind_v1 : std::uint8_t { production=1, multi_parent_derivation, exact_coverage, typed_port, generation, operation_compatibility };
struct planning_extension_contract_v1 { planning_extension_kind_v1 kind; std::string_view source_family; bool requires_exact_identity; bool requires_generation; };
inline constexpr std::array<planning_extension_contract_v1,6> typed_composition_contracts_v1{{
 {planning_extension_kind_v1::production,"grammar/explicit_production_registry_v1.hh",true,true},
 {planning_extension_kind_v1::multi_parent_derivation,"composition/derivation_dag_v1.hh",true,true},
 {planning_extension_kind_v1::exact_coverage,"grammar/exact_coverage_equation_v1.hh",true,true},
 {planning_extension_kind_v1::typed_port,"atom/port_v1.hh",true,false},
 {planning_extension_kind_v1::generation,"composition/value_plane_substitution_v1.hh",true,true},
 {planning_extension_kind_v1::operation_compatibility,"composition/parameter_binding_v1.hh",true,true},
}};
struct composition_validation_v1 { bool typed_symbols=false, acyclic_derivation=false, exact_coverage=false, ports_compatible=false, generations_fresh=false, operations_compatible=false; };
[[nodiscard]] constexpr bool valid_composition(composition_validation_v1 v) noexcept {return v.typed_symbols&&v.acyclic_derivation&&v.exact_coverage&&v.ports_compatible&&v.generations_fresh&&v.operations_compatible;}
} // namespace Cellerator::compiler::migration
