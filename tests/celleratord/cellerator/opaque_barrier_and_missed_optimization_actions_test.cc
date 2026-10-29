#include "../../../src/compiler/tooling/cellerator/tooling_model.hh"
#include <cassert>
using namespace cellerator::compiler::tooling::v1;int main(){auto a=missed_optimization_actions("native unknown_profile canonical");assert(a.size()==3&&a[0].safe&&a[1].fix_it.find("persistent")!=a[1].fix_it.npos&&a[2].canonicalization_cost==240);auto fixed=missed_optimization_actions("effects(reads values) persistent(structure) execution_order");assert(fixed.empty());}
