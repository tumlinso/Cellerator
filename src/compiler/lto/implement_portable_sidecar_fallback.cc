#include <Cellerator/compiler/lto/implement_portable_sidecar_fallback_v1.hh>
#include <iomanip>
#include <sstream>
namespace cellerator::compiler::lto::v1 {
content_identity_v1 identify_sidecar_content_v1(const std::vector<std::uint8_t>&p)noexcept{content_identity_v1 h{};std::uint64_t x=1469598103934665603ull;for(auto b:p){x^=b;x*=1099511628211ull;}for(std::size_t i=0;i<h.size();++i){x^=x>>12;x^=x<<25;x^=x>>27;h[i]=static_cast<std::uint8_t>(x); }return h;}
std::optional<std::size_t> resolve_sidecar_v1(const object_sidecar_reference_v1&r,const std::vector<ceir_sidecar_v1>&c)noexcept{for(std::size_t i=0;i<c.size();++i)if(c[i].identity==r.identity&&identify_sidecar_content_v1(c[i].payload)==r.identity)return i;return{};}
std::string sidecar_filename_v1(content_identity_v1 h){std::ostringstream o;o<<"ceir-";for(std::size_t i=0;i<8;++i)o<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(h[i]);return o.str()+".ceir";}
}
