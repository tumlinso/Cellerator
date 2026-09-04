#include "../../../src/compiler/tooling/cellerator/tooling_model.hh"
#include <cassert>
using namespace cellerator::compiler::tooling::v1;int main(){auto v=query_generations("value write"),s=query_generations("structure edit");assert(v.value==2&&v.structure==1&&v.stale_artifacts[0]=="value-binding");assert(s.structure==2&&s.support==2&&s.order==2&&s.stale_artifacts.size()==3);}
