#include <Cellerator/compiler/pass/cache_compiled_transforms_safely_v1.hh>

#include <cassert>
#include <filesystem>
#include <fstream>

namespace cp = cellerator::compiler::pass::v1;

namespace {
bool build(const std::string& path, void* data) noexcept {
    ++*static_cast<int*>(data);
    std::ofstream output(path);
    output << "artifact";
    return static_cast<bool>(output);
}
}

int main() {
    const auto directory = std::filesystem::temp_directory_path() / "ce_ccp1_g02_014";
    std::filesystem::remove_all(directory);
    int builds = 0;
    cp::transform_cache_request_v1 request{{"source-v1", "api-v1", "abi-v1",
        "gcc-15", "x86_64-linux", {"dep-a@1", "dep-b@2"}, "trusted"},
        directory.string(), false, build, &builds};
    const auto cold = cp::get_or_build_cached_transform_v1(request);
    const auto warm = cp::get_or_build_cached_transform_v1(request);
    assert(cold.status == cp::transform_cache_status_v1::success && !cold.warm_hit);
    assert(warm.status == cp::transform_cache_status_v1::success && warm.warm_hit);
    assert(builds == 1 && cold.artifact_path == warm.artifact_path);
    request.identity.dependency_identities[1] = "dep-b@3";
    const auto invalidated = cp::get_or_build_cached_transform_v1(request);
    assert(!invalidated.warm_hit && builds == 2);
    assert(cold.elapsed_nanoseconds > 0 && warm.elapsed_nanoseconds > 0);
    std::filesystem::remove_all(directory);
}
