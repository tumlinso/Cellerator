#include "../../../src/compiler/tooling/cellerator/tooling_model.hh"
#include <cassert>
using namespace cellerator::compiler::tooling::v1;int main(){auto a=planning_ir_at_cursor(false);assert(a.atom_proposals.size()==2&&a.candidates.size()==2&&a.candidates[0].cost==120&&a.candidates[0].selected&&a.candidates[1].rejected_reason=="dominated");auto f=planning_ir_at_cursor(true);assert(f.candidates[1].selected&&f.candidates[1].forced);}
