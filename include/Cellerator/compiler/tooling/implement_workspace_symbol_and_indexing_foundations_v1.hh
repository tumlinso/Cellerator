#pragma once
#include <string>
#include <string_view>
#include <vector>
namespace Cellerator::compiler::tooling {
enum class workspace_symbol_kind_v1 { cxx, domain, relation, field, profile, compiler_pass, ir_symbol };
struct workspace_symbol_v1 { std::string name,root,translation_unit,fingerprint; workspace_symbol_kind_v1 kind=workspace_symbol_kind_v1::cxx; bool exported=false; };
struct index_update_v1 { std::string root,translation_unit,fingerprint; std::vector<workspace_symbol_v1> symbols; };
class workspace_symbol_index_v1 {
public: bool update(index_update_v1 update); std::vector<workspace_symbol_v1> find(std::string_view name) const;
 std::size_t size()const noexcept{return symbols_.size();}
private: std::vector<workspace_symbol_v1> symbols_;
};
} // namespace Cellerator::compiler::tooling
