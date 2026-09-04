#include <Cellerator/compiler/program/import_cross_operation_rewrite_and_fusion_search_v1.hh>
#include <cassert>
using namespace Cellerator::compiler::composition;
int main(){connected_rewrite_v1 a{"share","edges","packed","state","sum-tree",false,false,true,10,7};auto fused=a;fused.name="fuse";fused.fusion=true;fused.field_authorized=true;fused.rewritten_cost=6;auto denied=fused;denied.name="denied";denied.field_authorized=false;auto slow=a;slow.name="slow";slow.rewritten_cost=11;auto r=select_connected_rewrites_v1({a,fused,denied,slow});assert(r.size()==2&&r[0].name=="share"&&r[1].name=="fuse");}
