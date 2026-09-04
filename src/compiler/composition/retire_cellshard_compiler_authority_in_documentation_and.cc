#include <Cellerator/compiler/composition/retire_cellshard_compiler_authority_in_documentation_and_v1.hh>
namespace Cellerator::compiler::composition {
std::vector<std::string> audit_retired_cellshard_compiler_authority_v1(const std::vector<compiler_source_entry_v1>&xs){std::vector<std::string>r;for(const auto&x:xs)if(x.path.find("CellShard/compiler")!=std::string::npos&&(!x.forwarding_alias||x.registers_compiler))r.push_back(x.path);return r;}
}
