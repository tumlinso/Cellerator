#pragma once
#include <Cellerator/compiler/reflection/implement_reflection_of_realization_ir_v1.hh>
#include <cstdint>
#include <string>
#include <vector>
namespace cellerator::compiler::reflection::v1 {
enum class inline_realization_validation_v1:std::uint8_t{verified=1,checked,unchecked};
struct typed_realization_binding_v1{std::string name,type;std::uint64_t identity=0,generation=0;};
struct inline_realization_block_v1{std::string backend;std::vector<std::string>projections,packs,stages,target_operations,native_fragments;std::vector<typed_realization_binding_v1>bindings;inline_realization_validation_v1 validation=inline_realization_validation_v1::verified;bool unsafe_acknowledged=false;};
enum class inline_realization_status_v1:std::uint8_t{valid=0,missing_backend,missing_stage,invalid_binding,unchecked_not_acknowledged};
[[nodiscard]] inline_realization_status_v1 validate_inline_realization_block_v1(const inline_realization_block_v1&)noexcept;
[[nodiscard]] reflected_realization_v1 override_realization_stage_v1(const reflected_realization_v1&,const inline_realization_block_v1&,std::size_t stage_index);
}
