#include <Cellerator/compiler/program/port_composition_basis_graph_and_schedule_tests_v1.hh>
#include <cassert>
using namespace Cellerator::compiler::composition;
int main(){std::vector<ported_test_inventory_v1>x={{"legacy/derivation","e03/derivation","sha1",ported_test_kind_v1::derivation,4,4},{"legacy/no_basis","e03/no_basis","sha2",ported_test_kind_v1::no_basis,2,2},{"legacy/coverage","e03/coverage","sha3",ported_test_kind_v1::exact_coverage,5,5},{"legacy/perf","e03/perf","sha4",ported_test_kind_v1::performance_baseline,3,3}};std::string e;assert(reconcile_ported_test_inventory_v1(x,&e));auto bad=x;bad[0].ported_cases=3;assert(!reconcile_ported_test_inventory_v1(bad,&e));x.pop_back();assert(!reconcile_ported_test_inventory_v1(x,&e));}
