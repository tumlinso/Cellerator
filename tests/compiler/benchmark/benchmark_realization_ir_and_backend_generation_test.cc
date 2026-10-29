#include <array>
#include <cassert>
#include <string_view>
int main(){ constexpr std::array<std::string_view,4> backends{"cpu","nvcc","clang_cuda","direct_ptx"}; constexpr std::array<std::string_view,9> metrics{"realization","projection_plan","packing_plan","stage_build","source_bytes","compiler_time","ptxas_resources","artifact_bytes","provenance"}; static_assert(backends.size()*metrics.size()==36); assert(backends.front()=="cpu"); }
