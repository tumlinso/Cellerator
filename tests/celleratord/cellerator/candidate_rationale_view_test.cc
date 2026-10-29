#include "../../../src/compiler/tooling/cellerator/tooling_model.hh"
#include <cassert>
using namespace cellerator::compiler::tooling::v1;int main(){for(auto k:{"analytical","measured","cached","external"}){auto e=explain_candidate(k,false);assert(e.evidence_kind==k&&e.complete_cost==120&&!e.freshness.empty()&&!e.uncertainty.empty()&&!e.fallback.empty());}auto f=explain_candidate("measured",true);assert(f.complete_cost==300&&f.user_edits=="forced by user");}
