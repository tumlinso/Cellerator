#include <Cellerator/compiler/lto/implement_template_instantiation_deduplication_v1.hh>
#include <cassert>
using namespace cellerator::compiler::lto::v1;
int main(){template_instantiation_v1 f{"apply","float","genes","base","cuda","a",9},f2=f,h=f;f2.symbol="b";h.numeric_type="half";h.symbol="h";template_deduplication_v1 r;assert(deduplicate_template_instantiations_v1({f,f2,h},&r)==template_dedup_status_v1::valid&&r.canonical.size()==2&&r.canonical_for_input[0]==r.canonical_for_input[1]);auto d=h;d.numeric_type="double";d.symbol="d";assert(deduplicate_template_instantiations_v1({f,f2,h,d},&r)==template_dedup_status_v1::valid&&r.canonical.size()==3);f2.body_hash=10;assert(deduplicate_template_instantiations_v1({f,f2},&r)==template_dedup_status_v1::odr_conflict);}
