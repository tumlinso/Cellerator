#include <Cellerator/compiler/migration/define_temporary_compiler_to_cellshard_migration_adapter_v1.hh>
#include <iostream>
#include <set>
#include <stdexcept>
using namespace Cellerator::compiler::migration;
int main(){try{std::set<std::string_view> names;for(auto a:temporary_adapters_v1){if(!a.version||a.owns_semantics||a.retirement_proof.empty()||a.target_cellerator_contract.rfind("Cellerator::compiler",0)!=0||!names.insert(a.legacy_surface).second)throw std::runtime_error("invalid temporary adapter");}if(names.size()!=5)throw std::runtime_error("missing adapter family");std::cout<<"validated 5 removable non-semantic adapters\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
