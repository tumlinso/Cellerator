#include <Cellerator/compiler/program/import_workload_family_representation_v1.hh>
#include <cassert>
using namespace Cellerator::compiler::composition;
int main(){workload_family_v1 f{"semantic.relation_apply","profile.pbmc","sm70",100,mutation_horizon_v1::per_generation,{{"latency",1},{"memory",.2}}};std::string e;assert(validate_workload_family_v1(f,&e));auto path=f;path.profile_family="/data/profile";assert(!validate_workload_family_v1(path,&e));auto empty=f;empty.objectives.clear();assert(!validate_workload_family_v1(empty,&e));}
