#pragma once
#include <Cellerator/compiler/reflection/freeze_the_compile_time_ir_handle_model_v1.hh>
#include <cstdint>
#include <string>
namespace cellerator::compiler::reflection::v1 {
enum class binding_source_kind_v1:std::uint8_t{source_variable=1,cpp_expression,ceir_value,profile_state,runtime_slot,generated_symbol};
enum class binding_semantics_v1:std::uint8_t{alias=1,move,reference,value};
struct typed_capture_binding_v1{std::string name,type;binding_source_kind_v1 source=binding_source_kind_v1::source_variable;binding_semantics_v1 semantics=binding_semantics_v1::reference;ir_handle_v1 handle{};const void*runtime_address=nullptr;std::uint64_t value_generation=0;handle_lifetime_v1 lifetime=handle_lifetime_v1::expression;};
enum class typed_binding_status_v1:std::uint8_t{valid=0,missing_name,missing_type,missing_source,ambiguous_lifetime,stale_generation,illegal_move};
[[nodiscard]] typed_binding_status_v1 validate_typed_binding_v1(const typed_capture_binding_v1&,std::uint64_t expected_generation)noexcept;
}
