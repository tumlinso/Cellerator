#include <Cellerator/compiler/diagnostics/implement_source_to_native_maps_and_removable_sections_v1.hh>
#include <cassert>
#include <array>
int main(){using namespace cellerator::compiler::diagnostics::v1;for(auto s:std::array{provenance_storage::sidecar,provenance_storage::object_debug_section,provenance_storage::separate_debug_file}){provenance_image i{4096,s,{{1,8,32},{2,8,64}}};assert(valid_source_native_map(i));auto stripped=strip_provenance(i);assert(stripped.hot_bytes==i.hot_bytes&&stripped.cold_map.empty());}assert(!valid_source_native_map({4096,provenance_storage::sidecar,{{0,1,0}}}));}
