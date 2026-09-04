#include <Cellerator/compiler/ir/realization/implement_generation_readiness_and_publication_v1.hh>
#include <cassert>
using namespace cellerator::compiler::ir::realization::v1;
int main(){generation_publication_v1 p{4,5,{{1,1},order_class_v1::persistent_physical},{{{2,1},component_readiness_v1::ready,5},{{2,2},component_readiness_v1::preparing,5}},false,false};assert(validate_generation_publication_v1(p)==generation_publication_status_v1::component_pending);assert(publish_generation_v1(&p)==generation_publication_status_v1::component_pending);p.components[1].state=component_readiness_v1::ready;assert(publish_generation_v1(&p)==generation_publication_status_v1::ready);assert(p.current_generation==5&&p.pending_generation==0&&p.retained_order.order==order_class_v1::persistent_physical);}
