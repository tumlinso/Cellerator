#include <Cellerator/math/matrix/patch.hh>
// Reuse the existing mathematical owner, compiled unchanged from the completed
// native patch provider. These private C symbols are not the public contract.
extern "C" int patch_forward(int,const float*,const float*,const float*,float*,float*,float*);
extern "C" int patch_vjp(int,const float*,const float*,const float*,const float*,const float*,const float*,float*,float*,float*);
extern "C" int patch_jvp(int,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,float*);
namespace cellerator::math::matrix {
namespace {
status admit(const patch_descriptor& d,const patch_primal& p,std::uint64_t& count) {
    if(!valid_axis(d.input_rows) || !valid_axis(d.input_columns) || !valid_axis(d.output_rows) || !valid_axis(d.output_columns)) return status::invalid_axes;
    const auto n=d.input_rows.extent;
    if(!n || n>46340 || d.input_columns.extent!=n || d.output_rows.extent!=n || d.output_columns.extent!=n
        || (d.row_stride && d.row_stride!=n) || !valid_policy(d.policy)) return status::unsupported_policy;
    count=n*n;
    if(p.state.size()!=count || p.left.size()!=count || p.right.size()!=count) return status::invalid_binding;
    return validate_generation(p.current);
}
status admit_tape(const patch_tape& t,std::uint64_t& count) {
    auto s=admit(t.descriptor,t.primal,count); if(s!=status::success) return s;
    if(!same_generations(*t.primal.current,t.saved)) return status::stale_generation;
    if(t.preactivation.size()!=count || t.activation.size()!=count) return status::invalid_binding;
    return status::success;
}
}
status patch_forward(const patch_descriptor& d,const patch_primal& p,patch_workspace workspace,
    std::span<float> output,patch_tape& tape) noexcept {
    std::uint64_t n=0; auto s=admit(d,p,n); if(s!=status::success) return s;
    if(workspace.preactivation.size()!=n || workspace.activation.size()!=n || output.size()!=n) return status::invalid_binding;
    s=validate_buffers({p.state,p.left,p.right},{workspace.preactivation,workspace.activation,output}); if(s!=status::success) return s;
    s=validate_metadata(std::span(p.current,1),{workspace.preactivation,workspace.activation,output}); if(s!=status::success) return s;
    if(::patch_forward(static_cast<int>(d.input_rows.extent),p.state.data(),p.left.data(),p.right.data(),
        workspace.preactivation.data(),workspace.activation.data(),output.data())) return status::provider_failure;
    tape={d,p,*p.current,workspace.preactivation,workspace.activation}; return status::success;
}
status patch_vjp(const patch_tape& t,std::span<const float> g,std::span<float> dx,std::span<float> dl,std::span<float> dr) noexcept {
    std::uint64_t n=0; auto s=admit_tape(t,n); if(s!=status::success) return s;
    if(g.size()!=n || dx.size()!=n || dl.size()!=n || dr.size()!=n) return status::invalid_binding;
    s=validate_buffers({t.primal.state,t.primal.left,t.primal.right,t.preactivation,t.activation,g},{dx,dl,dr}); if(s!=status::success) return s;
    s=validate_metadata(std::span(t.primal.current,1),{dx,dl,dr}); if(s!=status::success) return s;
    return ::patch_vjp(static_cast<int>(t.descriptor.input_rows.extent),t.primal.state.data(),t.primal.left.data(),t.primal.right.data(),
        t.preactivation.data(),t.activation.data(),g.data(),dx.data(),dl.data(),dr.data()) ? status::provider_failure:status::success;
}
status patch_jvp(const patch_tape& t,std::span<const float> dx,std::span<const float> dl,std::span<const float> dr,std::span<float> output) noexcept {
    std::uint64_t n=0; auto s=admit_tape(t,n); if(s!=status::success) return s;
    if(dx.size()!=n || dl.size()!=n || dr.size()!=n || output.size()!=n) return status::invalid_binding;
    s=validate_buffers({t.primal.state,t.primal.left,t.primal.right,t.preactivation,t.activation,dx,dl,dr},{output}); if(s!=status::success) return s;
    s=validate_metadata(std::span(t.primal.current,1),{output}); if(s!=status::success) return s;
    return ::patch_jvp(static_cast<int>(t.descriptor.input_rows.extent),t.primal.state.data(),t.primal.left.data(),t.primal.right.data(),
        t.preactivation.data(),t.activation.data(),dx.data(),dl.data(),dr.data(),output.data()) ? status::provider_failure:status::success;
}
} // namespace cellerator::math::matrix
