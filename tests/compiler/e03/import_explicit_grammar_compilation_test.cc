#include <Cellerator/compiler/program/import_explicit_grammar_compilation_v1.hh>
#include <cassert>
using namespace Cellerator::compiler::composition;
int main(){typed_production_contract_v1 p{"join",{{"a","atom",production_value_role_v1::input,"gene","o",1},{"b","relation",production_value_role_v1::output,"gene","o",2}},{{"b",{{"a",1}},true}},{production_effect_v1::pure},{},"id","order","gen","verify"};auto x=compile_explicit_grammar_v1({p},{{"A","atom",true}});assert(x.valid&&x.derivations.size()==1&&x.derivations[0].production=="join");auto bad=compile_explicit_grammar_v1({p},{{"A","atom",false}});assert(!bad.valid&&bad.derivations.empty());}
