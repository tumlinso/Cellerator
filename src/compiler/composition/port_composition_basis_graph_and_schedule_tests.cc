#include <Cellerator/compiler/composition/port_composition_basis_graph_and_schedule_tests_v1.hh>
#include <array>
namespace Cellerator::compiler::composition {
bool reconcile_ported_test_inventory_v1(const std::vector<ported_test_inventory_v1>&xs,std::string*e){std::array<bool,4> kinds{};for(const auto&x:xs){if(x.source_fixture.empty()||x.ported_fixture.empty()||x.provenance_hash.empty()||x.source_cases!=x.ported_cases){if(e)*e="inventory mismatch: "+x.source_fixture;return false;}kinds[static_cast<std::size_t>(x.kind)]=true;}for(bool k:kinds)if(!k){if(e)*e="missing required test category";return false;}return true;}
}
