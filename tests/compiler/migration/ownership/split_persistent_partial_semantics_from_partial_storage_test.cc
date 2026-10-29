#include <Cellerator/compiler/migration/split_persistent_partial_semantics_from_partial_storage_v1.hh>
#include <iostream>
#include <set>
#include <stdexcept>
using namespace Cellerator::compiler::migration;
int main(){try{std::set<std::string_view> concerns;std::size_t ce=0,cs=0;for(auto row:partial_interface_v1){if(!concerns.insert(row.concern).second)throw std::runtime_error("duplicate concern");row.owner==partial_owner_v1::cellerator?++ce:++cs;}if(ce!=4||cs!=4||!acyclic_partial_dependency_v1())throw std::runtime_error("cyclic or incomplete partial split");std::cout<<"validated acyclic 4+4 partial interface split\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
