#include <Cellerator/compiler/lto/assign_cross_tu_field_and_symbol_identities_v1.hh>
#include <cassert>
using namespace cellerator::compiler::lto::v1;
int main(){cross_tu_symbol_v1 a{exported_entity_kind_v1::field,"x","a",{1,1},linkage_v1::weak},b=a;b.module="b";std::vector<resolved_cross_tu_symbol_v1>o;assert(assign_cross_tu_identities_v1({a,b},&o)==cross_tu_identity_status_v1::valid&&o[0].identity.high==o[1].identity.high);b.semantic_fingerprint.low=2;assert(assign_cross_tu_identities_v1({a,b},&o)==cross_tu_identity_status_v1::odr_conflict);a.linkage=b.linkage=linkage_v1::hidden;b.semantic_fingerprint=a.semantic_fingerprint;assert(assign_cross_tu_identities_v1({a,b},&o)==cross_tu_identity_status_v1::valid&&o[0].identity.high!=o[1].identity.high);}
