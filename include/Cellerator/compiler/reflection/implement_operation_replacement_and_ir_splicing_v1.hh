#pragma once
#include <Cellerator/compiler/reflection/implement_reflection_of_operations_and_relations_v1.hh>
#include <cstdint>
#include <string>
#include <vector>
namespace cellerator::compiler::reflection::v1 {
enum class splice_kind_v1:std::uint8_t{before=1,after,replace,wrap,inline_region};
struct result_substitution_v1{std::uint32_t old_result=0,new_result=0;};
struct operation_splice_v1{splice_kind_v1 kind=splice_kind_v1::replace;std::uint32_t anchor=0;reflected_operation_v1 operation{};std::vector<std::uint32_t>operand_definitions;std::vector<result_substitution_v1>results;std::string profile_state;bool preserve_source_identity=true;};
enum class splice_status_v1:std::uint8_t{valid=0,invalid_anchor,dominance_violation,effect_violation,identity_violation,profile_missing,result_mismatch};
[[nodiscard]] splice_status_v1 validate_operation_splice_v1(const std::vector<reflected_operation_v1>&,const operation_splice_v1&)noexcept;
[[nodiscard]] std::vector<reflected_operation_v1> apply_operation_splice_v1(const std::vector<reflected_operation_v1>&,const operation_splice_v1&);
}
