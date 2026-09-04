#include <Cellerator/compiler/lto/implement_portable_sidecar_fallback_v1.hh>
#include <cassert>
using namespace cellerator::compiler::lto::v1;
int main(){ceir_sidecar_v1 s;s.payload={1,2,3,4};s.identity=identify_sidecar_content_v1(s.payload);object_sidecar_reference_v1 r{s.identity,"old/build/path"};std::vector<ceir_sidecar_v1>moved{{{},{}},s};assert(resolve_sidecar_v1(r,moved)==1);r.hint="new/install/path";assert(resolve_sidecar_v1(r,moved)==1);assert(sidecar_filename_v1(s.identity).rfind("ceir-",0)==0);moved[1].payload[0]=9;assert(!resolve_sidecar_v1(r,moved));}
