#include <Cellerator/compiler/discovery/import_relation_motif_and_operation_trace_discovery_v1.hh>
#include <cassert>
using namespace Cellerator::compiler::discovery;
int main(){
 operation_trace_event_v1 a{{1,1},{1,2},{1,3},{1,4},{1,5},{1,6},{1,7}};
 auto b=a;b.operation={1,8};b.relation={1,9};
 std::vector<trace_motif_v1> motifs;
 assert(discover_relation_and_trace_motifs_v1({a,b,a,b},2,2,&motifs)==trace_discovery_status_v1::success);
 assert(!motifs.empty()&&motifs[0].sequence_length==2&&motifs[0].occurrence_count==2);
 assert(motifs[0].sequence[0].numeric_policy==a.numeric_policy);
 auto changed=a;changed.profile={};
 assert(discover_relation_and_trace_motifs_v1({a,changed},2,2,&motifs)==trace_discovery_status_v1::invalid_event);
}
