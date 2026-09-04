#include <Cellerator/compiler/migration/define_cellerator_ownership_of_atom_semantics_v1.hh>
#include <Cellerator/compiler/migration/define_cellerator_ownership_of_basis_selection_v1.hh>
#include <Cellerator/compiler/migration/define_cellerator_ownership_of_evidence_and_proposal_dis_v1.hh>
#include <Cellerator/compiler/migration/define_cellerator_ownership_of_exact_certification_v1.hh>
#include <Cellerator/compiler/migration/define_cellerator_ownership_of_global_operation_program_v1.hh>
#include <Cellerator/compiler/migration/define_cellerator_ownership_of_superatom_promotion_v1.hh>
#include <Cellerator/compiler/migration/define_cellerator_ownership_of_typed_composition_and_gra_v1.hh>
#include <Cellerator/compiler/migration/define_cellerator_ownership_of_portable_schedule_compila_v1.hh>
#include <Cellerator/compiler/migration/define_cellerator_ruleset_export_consumed_by_cellshard_v1.hh>
#include <Cellerator/compiler/migration/define_temporary_compiler_to_cellshard_migration_adapter_v1.hh>
#include <Cellerator/compiler/migration/define_the_retained_cellshard_concrete_application_bound_v1.hh>
#include <Cellerator/compiler/migration/freeze_the_compiler_ownership_rehoming_contract_v1.hh>
#include <Cellerator/compiler/migration/reconcile_old_jbc_documentation_and_active_run_status_v1.hh>
#include <Cellerator/compiler/migration/split_persistent_partial_semantics_from_partial_storage_v1.hh>
#include <iostream>
#include <set>
#include <stdexcept>
using namespace Cellerator::compiler::migration;
int main(){try{std::set<std::string_view> old;for(auto r:compiler_ownership_rehoming_v1)if(!old.insert(r.old_cellshard_family).second)throw std::runtime_error("duplicate subsystem");if(old.size()!=12||!complete_ownership_map_v1()||!application_only_boundary_v1()||!acyclic_partial_dependency_v1()||!preserves_history_v1())throw std::runtime_error("ownership contract incomplete");for(auto a:temporary_adapters_v1)if(a.owns_semantics)throw std::runtime_error("adapter owns semantics");if(owner_of(atom_level_v1::resident)!=atom_owner_v1::cellshard_application)throw std::runtime_error("resident instance owner wrong");if(authorizes_execution(proposal_evidence_identity_v1{1,2,3,4,5}))throw std::runtime_error("proposal certifies execution");std::cout<<"validated 12-row complete compiler ownership map\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
