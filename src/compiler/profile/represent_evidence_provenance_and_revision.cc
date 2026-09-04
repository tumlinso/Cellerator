#include <Cellerator/compiler/profile/represent_evidence_provenance_and_revision_v1.hh>
#include <cmath>
namespace cellerator::compiler::profile::v1 {
namespace { bool zero(profile_identity_v1 x){return x.low==0u&&x.high==0u;} void mix(std::uint64_t& h,std::uint64_t x){h^=x+0x9e3779b97f4a7c15ull+(h<<6u)+(h>>2u);} }
evidence_provenance_status_v1 validate_evidence_provenance_v1(const evidence_provenance_v1& p) noexcept {
 if(p.schema_version!=evidence_provenance_schema_version_v1)return evidence_provenance_status_v1::unsupported_schema;
 if(zero(p.evidence)||zero(p.semantic_subject)||zero(p.dataset)||zero(p.source)||zero(p.producer)||zero(p.tool_version))return evidence_provenance_status_v1::invalid_identity;
 if(p.window_end<p.window_begin)return evidence_provenance_status_v1::invalid_window;
 if(!std::isfinite(p.confidence)||p.confidence<0.0||p.confidence>1.0)return evidence_provenance_status_v1::invalid_confidence;
 return evidence_provenance_status_v1::ok;
}
std::uint64_t evidence_cache_identity_v1(const evidence_provenance_v1& p) noexcept {
 std::uint64_t h=1469598103934665603ull; const profile_identity_v1 ids[]={p.evidence,p.dataset,p.source,p.sampling_method,p.transformation_stage,p.producer,p.tool_version,p.validity_predicate_set};
 for(auto id:ids){mix(h,id.low);mix(h,id.high);} mix(h,p.window_begin);mix(h,p.window_end);mix(h,p.evidence_revision);mix(h,p.validity_predicate_count); return h;
}
bool evidence_cache_compatible_v1(const evidence_provenance_v1&a,const evidence_provenance_v1&b) noexcept {return a.semantic_subject.low==b.semantic_subject.low&&a.semantic_subject.high==b.semantic_subject.high&&evidence_cache_identity_v1(a)==evidence_cache_identity_v1(b);}
}  // namespace cellerator::compiler::profile::v1
