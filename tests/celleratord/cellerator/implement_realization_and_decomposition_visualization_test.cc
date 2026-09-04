#include "../../../src/compiler/tooling/cellerator/tooling_model.hh"
#include <cassert>
using namespace cellerator::compiler::tooling::v1;int main(){auto v=realization_at_cursor();auto a=render_realization_json(v),b=render_realization_json(v);assert(a==b&&a.find("8192")!=a.npos&&v.atoms.size()==2&&v.stages.size()==3&&v.graph_capture&&!v.readiness.empty());}
