#pragma once
#include <string>
#include <vector>
namespace Cellerator::compiler::composition {
inline constexpr const char* compiler_authority_statement_v1="Cellerator owns composition, basis, graph, and schedule compilation; CellShard consumes narrow materialization requests.";
struct compiler_source_entry_v1{std::string path;bool forwarding_alias=false,registers_compiler=false;};
[[nodiscard]] std::vector<std::string> audit_retired_cellshard_compiler_authority_v1(const std::vector<compiler_source_entry_v1>&);
} // namespace Cellerator::compiler::composition
