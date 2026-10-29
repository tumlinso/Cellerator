#pragma once
#include <string>
#include <vector>
namespace Cellerator::compiler::composition {
struct derivation_parent_v1{std::string node;std::string production;bool alternative=false;};
struct derivation_node_v1{std::string id,type,lineage,reconstruction;std::vector<derivation_parent_v1> parents;};
struct derivation_dag_v1{std::vector<derivation_node_v1> nodes;};
struct derivation_validation_v1{bool valid=false;std::vector<std::string> topological_order;std::string diagnostic;};
[[nodiscard]] derivation_validation_v1 validate_derivation_dag_v1(const derivation_dag_v1&);
} // namespace Cellerator::compiler::composition
