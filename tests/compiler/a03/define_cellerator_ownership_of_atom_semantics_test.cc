#include <Cellerator/compiler/migration/define_cellerator_ownership_of_atom_semantics_v1.hh>
#include <array>
#include <iostream>
#include <set>
#include <stdexcept>
using namespace Cellerator::compiler::migration;
int main(){try{
 std::set<int> levels; std::size_t cellshard=0;
 for(auto row:atom_level_owners_v1){
  if(!levels.insert(static_cast<int>(row.level)).second) throw std::runtime_error("duplicate atom level");
  if(row.contract.empty()) throw std::runtime_error("missing atom behavior");
  if(row.owner==atom_owner_v1::cellshard_application) ++cellshard;
 }
 if(levels.size()!=8||cellshard!=1||owner_of(atom_level_v1::resident)!=atom_owner_v1::cellshard_application)
  throw std::runtime_error("atom ownership is not exhaustive and singular");
 for(auto level:{atom_level_v1::candidate,atom_level_v1::certified,atom_level_v1::basis,atom_level_v1::super,atom_level_v1::physical,atom_level_v1::replica,atom_level_v1::partial})
  if(owner_of(level)!=atom_owner_v1::cellerator_compiler) throw std::runtime_error("compiler state escaped Cellerator");
 std::cout<<"validated 8 uniquely owned atom states\n"; return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
