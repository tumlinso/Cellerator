#pragma once
#include <string>
#include <vector>
namespace Cellerator::compiler::composition {
struct planning_operation_v1{std::string id,semantic_ir_operation,graph_family,effect,local_fragment;std::vector<std::string> input_atoms,output_atoms,profile_variants,rewrites;};
struct planning_operation_graph_v1{std::vector<planning_operation_v1> operations;};
[[nodiscard]] bool validate_planning_operation_graph_v1(const planning_operation_graph_v1&,std::string*error=nullptr);
} // namespace Cellerator::compiler::composition
