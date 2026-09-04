#pragma once
#include <Cellerator/compiler/reflection/implement_inline_planning_ir_blocks_v1.hh>
#include <Cellerator/compiler/reflection/implement_inline_semantic_ir_blocks_v1.hh>
#include <Cellerator/compiler/reflection/implement_operation_replacement_and_ir_splicing_v1.hh>
#include <Cellerator/compiler/reflection/implement_reflection_of_profile_environments_v1.hh>
#include <cstdint>
#include <string>
#include <vector>
namespace cellerator::compiler::reflection::v1 {
struct source_defined_rewrite_request_v1{std::string cell_source,required_profile;reflected_profile_environment_v1 profiles;reflected_operation_v1 original;inline_semantic_block_v1 semantic_rewrite;inline_planning_block_v1 planning_rewrite;std::vector<std::uint64_t>row_offsets,column_indices;std::vector<double>values,input;};
struct rewrite_provenance_trace_v1{ir_handle_v1 original_operation{},original_provenance{},rewritten_operation{},rewritten_provenance{};};
struct source_defined_rewrite_result_v1{rewrite_provenance_trace_v1 trace{};std::string selected_candidate;std::vector<double>cpu_output,native_output;std::vector<reflected_operation_v1>rewritten_graph;};
enum class source_defined_rewrite_status_v1:std::uint8_t{success=0,invalid_source,profile_unavailable,semantic_invalid,planning_invalid,relation_invalid,output_mismatch};
[[nodiscard]] source_defined_rewrite_status_v1 compile_source_defined_rewrite_v1(const source_defined_rewrite_request_v1&,source_defined_rewrite_result_v1*,std::string*error=nullptr);
}
