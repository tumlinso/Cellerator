#include <Cellerator/compiler/program/implement_no_basis_and_multiple_basis_outcomes_v1.hh>
#include <cassert>
using namespace Cellerator::compiler::composition;
int main(){assert(!select_basis_outcome_v1({},"p",10).use_basis);std::vector<basis_outcome_v1>x={{"a","p",9,true,false},{"b","q",2,true,false},{"c","p",7,true,true},{"bad","p",1,false,false}};auto p=select_basis_outcome_v1(x,"p",10);assert(p.use_basis&&p.selected->id=="c"&&p.reason=="selected external offer");assert(!select_basis_outcome_v1(x,"z",10).use_basis);assert(!select_basis_outcome_v1(x,"p",5).use_basis);}
