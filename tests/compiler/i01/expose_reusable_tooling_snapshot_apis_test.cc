#include <Cellerator/compiler/tooling/expose_reusable_tooling_snapshot_apis_v1.hh>
#include <cassert>
using namespace Cellerator::compiler::tooling;
int main(){tooling_snapshot_v1 s({3,"cell Gene",{{"Gene",5,9}},{"cell",0,4}}, {"ok"}});assert(s.revision()==3&&s.symbol_at(6)->name=="Gene"&&!s.symbol_at(9));auto copy=s;assert(copy.diagnostics()[0]=="ok");int calls=0;tooling_cancellation_v1 c;request_background_compile_v1([&](auto,auto){++calls;},"a.cell",c);c.cancel();request_background_compile_v1([&](auto,auto){++calls;},"a.cell",c);assert(calls==1);}
