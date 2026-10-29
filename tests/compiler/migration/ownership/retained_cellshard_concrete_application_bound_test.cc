#include <Cellerator/compiler/migration/retained_cellshard_application_boundary_v1.hh>
#include <iostream>
#include <set>
#include <stdexcept>
using namespace Cellerator::compiler::migration;
int main(){try{std::set<std::string_view> names;for(auto r:retained_cellshard_boundary_v1)if(r.name.empty()||r.may_decide_compiler_semantics||!names.insert(r.name).second)throw std::runtime_error("invalid retained concern");if(names.size()!=10||!application_only_boundary_v1())throw std::runtime_error("incomplete concrete boundary");std::cout<<"validated 10 application-only CellShard concerns\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
