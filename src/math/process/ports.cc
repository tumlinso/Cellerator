#include <Cellerator/math/process/ports.hh>
#include <limits>
extern "C" int ports_transport(int,std::int64_t,std::int64_t,std::int64_t,const std::int64_t*,const std::int64_t*,const std::int64_t*,
    const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,float*,float*,float*,float*) noexcept;
namespace cellerator::math::process {
namespace {
status admit(const port_descriptor& d,const port_primal& p) {
    if(!matrix::valid_axis(d.actors) || !matrix::valid_axis(d.ports) || !matrix::valid_axis(d.hidden) || !matrix::valid_axis(d.edges)) return status::invalid_axes;
    const auto n=d.actors.extent,ports=d.ports.extent,k=d.edges.extent;
    const auto limit=static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()/sizeof(float));
    if(!n || !ports || n>limit/ports || k>limit || !matrix::valid_policy(d.policy)) return status::unsupported_policy;
    if(d.widths.size()!=n || d.private_axes.size()!=n || d.source.size()!=k || d.destination.size()!=k
        || !d.widths.data() || !d.private_axes.data() || (k && (!d.source.data() || !d.destination.data()))) return status::invalid_binding;
    std::uint64_t total=0;
    for(std::size_t i=0;i<n;++i) {
        if(d.widths[i]<=0 || static_cast<std::uint64_t>(d.widths[i])>limit-total) return status::invalid_binding;
        if(!matrix::valid_axis(d.private_axes[i]) || d.private_axes[i].extent!=static_cast<std::uint64_t>(d.widths[i])) return status::invalid_axes;
        total+=static_cast<std::uint64_t>(d.widths[i]);
    }
    if(total>limit/ports || total!=d.hidden.extent || p.hidden.size()!=total || p.encoder.size()!=total*ports
        || p.decoder.size()!=total*ports || p.weights.size()!=k) return status::invalid_binding;
    for(std::size_t i=0;i<k;++i) if(d.source[i]<0 || d.destination[i]<0 || static_cast<std::uint64_t>(d.source[i])>=n
        || static_cast<std::uint64_t>(d.destination[i])>=n) return status::invalid_binding;
    return matrix::validate_generation(p.current);
}
status admit_tape(const port_tape& t) {
    auto s=admit(t.descriptor,t.primal); if(s!=status::success) return s;
    return matrix::same_generations(*t.primal.current,t.saved) ? status::success:status::stale_generation;
}
status metadata(const port_descriptor& d,const port_primal& p,std::initializer_list<std::span<float>> writes) {
    for(auto data:{d.widths,d.source,d.destination}) {
        auto s=matrix::validate_metadata(data,writes); if(s!=status::success) return s;
    }
    auto s=matrix::validate_metadata(d.private_axes,writes); if(s!=status::success) return s;
    return matrix::validate_metadata(std::span(p.current,1),writes);
}
int call(int mode,const port_descriptor& d,const port_primal& p,const float* a,const float* b,const float* c,const float* f,
    float* out,float* oe=nullptr,float* od=nullptr,float* ow=nullptr) {
    return ::ports_transport(mode,static_cast<std::int64_t>(d.actors.extent),static_cast<std::int64_t>(d.ports.extent),
        static_cast<std::int64_t>(d.edges.extent),d.widths.data(),d.source.data(),d.destination.data(),p.hidden.data(),p.encoder.data(),
        p.decoder.data(),p.weights.data(),a,b,c,f,out,oe,od,ow);
}
}
status port_forward(const port_descriptor& d,const port_primal& p,std::span<float> output,port_tape& tape) noexcept {
    auto s=admit(d,p); if(s!=status::success) return s;
    if(output.size()!=p.hidden.size()) return status::invalid_binding;
    s=matrix::validate_buffers({p.hidden,p.encoder,p.decoder,p.weights},{output}); if(s!=status::success) return s;
    s=metadata(d,p,{output}); if(s!=status::success) return s;
    if(call(0,d,p,nullptr,nullptr,nullptr,nullptr,output.data())) return status::provider_failure;
    tape={d,p,*p.current}; return status::success;
}
status port_vjp(const port_tape& t,std::span<const float> g,std::span<float> dh,std::span<float> de,std::span<float> dd,std::span<float> dw) noexcept {
    auto s=admit_tape(t); if(s!=status::success) return s;
    const auto& p=t.primal;
    if(g.size()!=p.hidden.size() || dh.size()!=p.hidden.size() || de.size()!=p.encoder.size() || dd.size()!=p.decoder.size()
        || dw.size()!=p.weights.size()) return status::invalid_binding;
    s=matrix::validate_buffers({p.hidden,p.encoder,p.decoder,p.weights,g},{dh,de,dd,dw}); if(s!=status::success) return s;
    s=metadata(t.descriptor,p,{dh,de,dd,dw}); if(s!=status::success) return s;
    return call(1,t.descriptor,p,g.data(),nullptr,nullptr,nullptr,dh.data(),de.data(),dd.data(),dw.data()) ? status::provider_failure:status::success;
}
status port_jvp(const port_tape& t,std::span<const float> dh,std::span<const float> de,std::span<const float> dd,std::span<const float> dw,std::span<float> output) noexcept {
    auto s=admit_tape(t); if(s!=status::success) return s;
    const auto& p=t.primal;
    if(dh.size()!=p.hidden.size() || de.size()!=p.encoder.size() || dd.size()!=p.decoder.size() || dw.size()!=p.weights.size()
        || output.size()!=p.hidden.size()) return status::invalid_binding;
    s=matrix::validate_buffers({p.hidden,p.encoder,p.decoder,p.weights,dh,de,dd,dw},{output}); if(s!=status::success) return s;
    s=metadata(t.descriptor,p,{output}); if(s!=status::success) return s;
    return call(2,t.descriptor,p,dh.data(),de.data(),dd.data(),dw.data(),output.data()) ? status::provider_failure:status::success;
}
} // namespace cellerator::math::process
