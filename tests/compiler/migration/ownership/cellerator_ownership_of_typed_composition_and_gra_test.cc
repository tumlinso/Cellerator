#include <Cellerator/compiler/migration/typed_composition_graph_ownership_contract_v1.hh>
#include <iostream>
#include <set>
#include <stdexcept>
using namespace Cellerator::compiler::migration;
int main(){try{
 std::set<int> kinds;
 for(auto row:typed_composition_contracts_v1){if(!kinds.insert(int(row.kind)).second||row.source_family.empty()||!row.requires_exact_identity)throw std::runtime_error("invalid extension mapping");}
 if(kinds.size()!=6)throw std::runtime_error("missing extension family");
 composition_validation_v1 all{true,true,true,true,true,true}; if(!valid_composition(all))throw std::runtime_error("complete composition rejected");
 const composition_validation_v1 missing[]={{false,true,true,true,true,true},{true,false,true,true,true,true},{true,true,false,true,true,true},{true,true,true,false,true,true},{true,true,true,true,false,true},{true,true,true,true,true,false}};
 for(auto value:missing)if(valid_composition(value))throw std::runtime_error("incomplete composition accepted");
 std::cout<<"validated 6 typed composition contracts\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
