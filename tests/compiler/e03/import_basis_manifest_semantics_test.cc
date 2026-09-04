#include <Cellerator/compiler/program/import_basis_manifest_semantics_v1.hh>
#include <cassert>
using namespace Cellerator::compiler::composition;
int main(){basis_manifest_v1 m{"basis.pbmc","evidence-42",4096,7,true,{1,.25},{{"a","p","required",0},{"b","q","alternate",.5}}};auto text=print_basis_manifest_v1(m);auto r=parse_basis_manifest_v1(text);assert(r&&r->id==m.id&&r->budget_bytes==4096&&r->objective_vector==m.objective_vector&&r->members.size()==2&&r->members[1].redundancy==.5);assert(print_basis_manifest_v1(*r)==text);assert(!parse_basis_manifest_v1("broken"));}
