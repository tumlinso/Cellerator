#include <Cellerator/compiler/program/import_induced_grammar_as_experimental_search_v1.hh>
#include <cassert>
using namespace Cellerator::compiler::composition;
int main(){std::vector<induced_production_candidate_v1> c={{"fast","trace","exact",.95,8,true},{"slow","trace","exact",.99,12,true},{"unsafe","trace","exact",.98,2,false}};auto r=search_induced_grammar_v1(c,2,10,.9);assert(r.evaluated.size()==2&&r.no_promotion);auto promoted=search_induced_grammar_v1(c,3,10,.9);assert(promoted.promoted.size()==1&&promoted.promoted[0].name=="fast");auto bounded=search_induced_grammar_v1(c,1,10,.9);assert(bounded.evaluated.size()==1&&bounded.no_promotion);}
