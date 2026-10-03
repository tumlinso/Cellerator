#pragma once
#include <Cellerator/compute/operation/native_foundation_contract.hh>
#include <Cellerator/compute/operation/relation_semantics.hh>
#include <array>
#include <cfenv>
#include <initializer_list>
#include <span>
namespace cellerator::math::matrix {
namespace ex=execution;
namespace rel=compute::relation;
namespace nf=compute::operation::nf1;
enum class status { success, invalid_axes, invalid_binding, stale_generation,
    unsupported_policy, alias, provider_failure };
struct generations {
    ex::structure_id structure{};
    ex::structure_epoch epoch{};
    std::array<ex::value_generation,4> operands{};
};
struct capabilities {
    std::uint32_t actions=nf::forward|nf::vjp|nf::jvp;
    ex::numeric_type storage=ex::numeric_type::f32;
    bool cuda=false, capture=false, derivative_allocation=true, forward_allocation=false;
};
inline bool same_generations(const generations& a,const generations& b) noexcept {
    if(!ex::same_identity(a.structure,b.structure) || a.epoch.value!=b.epoch.value) return false;
    for(std::size_t i=0;i<4;++i) if(a.operands[i].value!=b.operands[i].value) return false;
    return true;
}
inline status validate_generation(const generations* current) noexcept {
    if(!current || !ex::valid_identity(current->structure) || !current->epoch.value) return status::invalid_binding;
    for(auto generation:current->operands) if(!generation.value) return status::invalid_binding;
    return status::success;
}
inline bool valid_axis(const rel::axis_descriptor& axis) noexcept {
    return ex::validate_persistent_axis_identity(axis.identity)==ex::biological_validation_code::ok;
}
inline bool valid_policy(const rel::arithmetic_policy& policy) noexcept {
    return policy.relation_storage==ex::numeric_type::f32 && policy.input_storage==ex::numeric_type::f32
        && policy.multiply==ex::numeric_type::f32 && policy.accumulation==ex::numeric_type::f32
        && policy.output_storage==ex::numeric_type::f32 && !policy.permit_fma && !policy.permit_reassociation
        && policy.nonfinite==rel::nonfinite_policy::propagate && std::fegetround()==FE_TONEAREST;
}
inline rel::arithmetic_policy host_policy() noexcept {
    return {ex::numeric_type::f32,ex::numeric_type::f32,ex::numeric_type::f32,
        ex::numeric_type::f32,ex::numeric_type::f32,false,false,rel::nonfinite_policy::propagate};
}
inline bool overlaps(std::span<const float> a,std::span<const float> b) noexcept {
    if(a.empty() || b.empty()) return false;
    const auto x=reinterpret_cast<std::uintptr_t>(a.data()),y=reinterpret_cast<std::uintptr_t>(b.data());
    return x<=y ? y-x<a.size_bytes():x-y<b.size_bytes();
}
template<class T> inline status validate_metadata(std::span<const T> metadata,
    std::initializer_list<std::span<float>> writes) noexcept {
    const auto start=reinterpret_cast<std::uintptr_t>(metadata.data());
    for(auto w:writes) {
        if(metadata.empty() || w.empty()) continue;
        const auto destination=reinterpret_cast<std::uintptr_t>(w.data());
        if(start<=destination ? destination-start<metadata.size_bytes() : start-destination<w.size_bytes())
            return status::alias;
    }
    return status::success;
}
inline status validate_buffers(std::initializer_list<std::span<const float>> reads,
    std::initializer_list<std::span<float>> writes) noexcept {
    for(auto r:reads) if(r.size() && !r.data()) return status::invalid_binding;
    for(auto w:writes) if(w.size() && !w.data()) return status::invalid_binding;
    for(auto w:writes) for(auto r:reads) if(overlaps(w,r)) return status::alias;
    for(auto i=writes.begin();i!=writes.end();++i) for(auto j=writes.begin();j!=i;++j)
        if(overlaps(*i,*j)) return status::alias;
    return status::success;
}
} // namespace cellerator::math::matrix
