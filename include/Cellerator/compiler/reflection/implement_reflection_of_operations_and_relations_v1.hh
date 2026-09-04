#pragma once
#include <Cellerator/compiler/reflection/freeze_the_compile_time_ir_handle_model_v1.hh>
#include <cstdint>
#include <string>
#include <vector>
namespace cellerator::compiler::reflection::v1 {
enum class reflected_effect_v1:std::uint8_t{none=0,reads=1,writes=2,reads_writes=3};
struct reflected_value_v1{ir_handle_v1 handle{};std::string type;std::uint64_t domain=0,order=0,generation=0;};
struct reflected_relation_v1{ir_handle_v1 handle{};std::uint64_t source_domain=0,target_domain=0,order=0,structure_epoch=0,value_generation=0;};
struct reflected_operation_v1{ir_handle_v1 handle{};std::string normalized_kind;std::vector<reflected_value_v1>operands,results;reflected_relation_v1 relation{};reflected_effect_v1 effects=reflected_effect_v1::none;ir_handle_v1 provenance{};};
[[nodiscard]] bool validate_reflected_operation_v1(const reflected_operation_v1&)noexcept;
[[nodiscard]] std::string dump_reflected_operation_v1(const reflected_operation_v1&);
}
