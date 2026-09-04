#include <Cellerator/compiler/api/preserve_direct_runtime_execution_access_v1.hh>
#include <algorithm>
namespace cellerator::compiler::api::v1 {
bool is_direct_runtime_surface_v1(std::string_view name) noexcept {
    return std::find(direct_runtime_surfaces_v1.begin(), direct_runtime_surfaces_v1.end(), name) != direct_runtime_surfaces_v1.end();
}
}
