#include <Cellerator/compiler/program/import_persistent_partial_compiler_semantics_v1.hh>
#include <cassert>
using namespace Cellerator::compiler::composition;
int main(){persistent_partial_semantics_v1 p{"partial","rows[0:10]","sum","identity","fp32 deterministic",{"relation","values"},2,4,100,30,4};auto d=evaluate_persistent_partial_v1(p,2,4);assert(d.legal&&d.persist);assert(!evaluate_persistent_partial_v1(p,3,4).legal);auto cold=p;cold.expected_reuse=2;auto c=evaluate_persistent_partial_v1(cold,2,4);assert(c.legal&&!c.persist);auto incomplete=p;incomplete.merge_algebra.clear();assert(!evaluate_persistent_partial_v1(incomplete,2,4).legal);}
