#pragma once
#include <Cellerator/compiler/composition/import_typed_composition_production_contracts_v1.hh>
#include <string>
#include <vector>
namespace Cellerator::compiler::composition {
struct grammar_atom_v1{std::string name,type;bool certified=false;};
struct grammar_derivation_v1{std::string production;std::vector<std::string> inputs;std::string output_type;};
struct grammar_compilation_v1{bool valid=false;std::vector<grammar_derivation_v1> derivations;std::vector<std::string> diagnostics;};
[[nodiscard]] grammar_compilation_v1 compile_explicit_grammar_v1(const std::vector<typed_production_contract_v1>&,const std::vector<grammar_atom_v1>&);
} // namespace Cellerator::compiler::composition
