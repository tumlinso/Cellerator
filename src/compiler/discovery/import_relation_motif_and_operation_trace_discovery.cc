#include <Cellerator/compiler/discovery/import_relation_motif_and_operation_trace_discovery_v1.hh>
#include <algorithm>
#include <map>
namespace Cellerator::compiler::discovery {
namespace {
bool valid(const operation_trace_event_v1& e) { return valid_persistent_atom_identity_v1(e.relation)&&valid_persistent_atom_identity_v1(e.source_domain)&&valid_persistent_atom_identity_v1(e.destination_domain)&&valid_persistent_atom_identity_v1(e.operation)&&valid_persistent_atom_identity_v1(e.numeric_policy)&&valid_persistent_atom_identity_v1(e.profile)&&valid_persistent_atom_identity_v1(e.field); }
void hash_id(std::uint64_t& h,persistent_atom_identity_v1 x){ h=(h^x.producer_namespace)*1099511628211ULL;h=(h^x.local_identity)*1099511628211ULL; }
std::uint64_t event_hash(const operation_trace_event_v1&e){std::uint64_t h=14695981039346656037ULL;hash_id(h,e.relation);hash_id(h,e.source_domain);hash_id(h,e.destination_domain);hash_id(h,e.operation);hash_id(h,e.numeric_policy);hash_id(h,e.profile);hash_id(h,e.field);return h;}
bool same(const operation_trace_event_v1&a,const operation_trace_event_v1&b){return a.relation==b.relation&&a.source_domain==b.source_domain&&a.destination_domain==b.destination_domain&&a.operation==b.operation&&a.numeric_policy==b.numeric_policy&&a.profile==b.profile&&a.field==b.field;}
}
trace_discovery_status_v1 discover_relation_and_trace_motifs_v1(const std::vector<operation_trace_event_v1>& trace,std::uint32_t maximum_length,std::uint32_t minimum_occurrences,std::vector<trace_motif_v1>* output) noexcept {
 if(output==nullptr||maximum_length==0||minimum_occurrences<2)return trace_discovery_status_v1::invalid_limit;
 if(!std::all_of(trace.begin(),trace.end(),valid))return trace_discovery_status_v1::invalid_event;
 std::vector<trace_motif_v1> result;
 try { for(std::uint32_t n=1;n<=maximum_length&&n<=trace.size();++n){
  std::map<std::vector<std::uint64_t>,std::vector<std::size_t>> groups;
  for(std::size_t i=0;i+n<=trace.size();++i){std::vector<std::uint64_t> key;for(std::size_t j=0;j<n;++j)key.push_back(event_hash(trace[i+j]));groups[key].push_back(i);}
  for(const auto&g:groups)if(g.second.size()>=minimum_occurrences){const auto start=g.second.front();bool exact=true;for(auto pos:g.second)for(std::size_t j=0;j<n;++j)exact&=same(trace[start+j],trace[pos+j]);if(!exact)continue;std::uint64_t h=14695981039346656037ULL;for(auto v:g.first)h=(h^v)*1099511628211ULL;result.push_back({{0x43454d4f54494631ULL,h},n,static_cast<std::uint32_t>(g.second.size()),{trace.begin()+start,trace.begin()+start+n}});}
 }} catch(...){return trace_discovery_status_v1::invalid_limit;}
 std::sort(result.begin(),result.end(),[](const auto&a,const auto&b){return a.sequence_length>b.sequence_length||(a.sequence_length==b.sequence_length&&a.motif_identity.local_identity<b.motif_identity.local_identity);});*output=std::move(result);return trace_discovery_status_v1::success;
}
}  // namespace Cellerator::compiler::discovery
