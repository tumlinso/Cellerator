#include "../../../src/compiler/tooling/cellerator/tooling_model.hh"
#include <cassert>
using namespace cellerator::compiler::tooling::v1;int main(){for(auto l:{"semantic","planning","realization"}){auto a=assist_inline_ceir(l,"missing_type");assert(a.level==l&&!a.operations.empty()&&!a.types.empty()&&!a.attributes.empty()&&!a.profiles.empty()&&!a.candidates.empty()&&!a.namespaces.empty()&&!a.validation_modes.empty()&&!a.diagnostics.empty());}assert(assist_inline_ceir("realization","").instructions[0]=="mma.sync");}
