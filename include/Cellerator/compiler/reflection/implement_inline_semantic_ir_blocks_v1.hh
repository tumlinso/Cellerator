#pragma once
#include <Cellerator/compiler/reflection/implement_reflection_of_operations_and_relations_v1.hh>
#include <string>
#include <vector>
namespace cellerator::compiler::reflection::v1 {
struct inline_semantic_capture_v1{std::string name;reflected_value_v1 value{};};
struct inline_semantic_block_v1{std::vector<inline_semantic_capture_v1>captures,results;reflected_operation_v1 replacement{};std::string surrounding_field,profile_state;};
enum class inline_semantic_status_v1:std::uint8_t{valid=0,invalid_operation,missing_context,domain_mismatch,generation_mismatch,effect_mismatch};
[[nodiscard]] inline_semantic_status_v1 validate_inline_semantic_block_v1(const inline_semantic_block_v1&)noexcept;
}
