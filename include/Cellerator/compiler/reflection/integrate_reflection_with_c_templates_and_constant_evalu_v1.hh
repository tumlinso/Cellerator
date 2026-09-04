#pragma once
#include <Cellerator/compiler/reflection/freeze_the_compile_time_ir_handle_model_v1.hh>
#include <cstdint>
#include <type_traits>
namespace cellerator::compiler::reflection::v1 {
template<handle_kind_v1 Kind>struct typed_ir_handle_v1{ir_handle_v1 value;constexpr explicit typed_ir_handle_v1(ir_handle_v1 h):value(h){}constexpr bool correct_kind()const noexcept{return value.kind==Kind;}};
template<class T>struct is_compile_time_ir_handle_v1:std::false_type{};
template<handle_kind_v1 K>struct is_compile_time_ir_handle_v1<typed_ir_handle_v1<K>>:std::true_type{};
template<std::uint32_t Bits,bool Floating,std::uint64_t DomainExtent>struct reflected_numeric_property_v1{static constexpr std::uint32_t bits=Bits;static constexpr bool floating=Floating;static constexpr std::uint64_t domain_extent=DomainExtent;};
template<class P>using reflected_scalar_v1=std::conditional_t<P::floating&&(P::bits==32),float,std::conditional_t<P::floating,double,std::uint64_t>>;
template<class P>constexpr bool vectorizable_v1()noexcept{return P::domain_extent>0&&P::bits>0&&(128%P::bits)==0;}
void template_reflection_link_anchor_v1()noexcept;
}
