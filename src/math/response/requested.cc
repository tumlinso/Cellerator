#include <Cellerator/math/response/requested.hh>
namespace cellerator::math::response {
nf::status bind_custom_rule(const nf::compiled_block& block,nf::capability action,const void* owner,
    std::uint64_t stage,std::uint64_t candidate,std::uint32_t binding,ex::program::prepared_stage_v2& out) noexcept {
    if(!owner)return nf::status::invalid_binding;
    return nf::bind_compiled_stage(block,action,owner,stage,candidate,binding,out);
}
template class requested_scaled_tanh<float>;
template class requested_scaled_tanh<double>;
}
