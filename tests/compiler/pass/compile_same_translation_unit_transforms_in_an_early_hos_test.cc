#include <Cellerator/compiler/pass/compile_same_translation_unit_transforms_in_an_early_hos_v1.hh>

#include <cassert>
#include <dlfcn.h>
#include <filesystem>

namespace cp = cellerator::compiler::pass::v1;

int main() {
    const auto directory = std::filesystem::temp_directory_path() / "ce_ccp1_g02_012";
    const cp::early_host_transform_request_v1 request{
        "extern \"C\" int same_unit_transform(int value) { return value + 7; }\n",
        "g++", "cellerator-pass-api-v1", "", directory.string()};
    const auto receipt = cp::compile_early_host_transform_v1(request);
    assert(receipt.status == cp::early_host_transform_status_v1::success);
    assert(receipt.cache_key == cp::early_host_transform_key_v1(request));
    void* handle = dlopen(receipt.artifact_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    assert(handle != nullptr);
    auto transform = reinterpret_cast<int (*)(int)>(dlsym(handle, "same_unit_transform"));
    assert(transform != nullptr && transform(5) == 12);
    dlclose(handle);
    std::filesystem::remove_all(directory);
}
