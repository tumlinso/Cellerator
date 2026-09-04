#pragma once
#include <string>
#include <vector>
namespace Cellerator::compiler::composition {
struct connected_rewrite_v1{std::string name,traversal,persistent_order,common_output,partial_tree;bool fusion=false,field_authorized=false,exact=false;double original_cost=0,rewritten_cost=0;};
[[nodiscard]] std::vector<connected_rewrite_v1> select_connected_rewrites_v1(const std::vector<connected_rewrite_v1>&);
} // namespace Cellerator::compiler::composition
