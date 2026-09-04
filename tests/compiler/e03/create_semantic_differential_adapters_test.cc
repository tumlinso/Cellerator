#include <Cellerator/compiler/program/create_semantic_differential_adapters_v1.hh>
#include <cassert>
using namespace Cellerator::compiler::composition;
int main(){canonical_semantics_v1 old{{"coverage","a+b"},{"schema","legacy"},{"cost","10"}},now{{"cost","10"},{"schema","planning-ir-v1"},{"coverage","a+b"}};auto d=compare_canonical_semantics_v1(old,now,{{"schema","schema ownership improved"}});assert(d.equivalent&&d.intentional.size()==1&&d.unexpected.empty());now["coverage"]="a";auto bad=compare_canonical_semantics_v1(old,now,{{"schema","schema ownership improved"}});assert(!bad.equivalent&&bad.unexpected[0]=="coverage");}
