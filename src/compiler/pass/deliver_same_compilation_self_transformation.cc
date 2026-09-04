#include <Cellerator/compiler/pass/deliver_same_compilation_self_transformation_v1.hh>

#include <Cellerator/compiler/pass/cache_compiled_transforms_safely_v1.hh>
#include <Cellerator/compiler/pass/compile_same_translation_unit_transforms_in_an_early_hos_v1.hh>

#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>

namespace cellerator::compiler::pass::v1 {
namespace {
struct build_context { early_host_transform_request_v1 request; };
bool build_transform(const std::string& output, void* data) noexcept {
    auto& context = *static_cast<build_context*>(data);
    const auto compiled = compile_early_host_transform_v1(context.request);
    if (compiled.status != early_host_transform_status_v1::success) return false;
    try {
        std::filesystem::copy_file(compiled.artifact_path, output,
            std::filesystem::copy_options::overwrite_existing);
    } catch (...) { return false; }
    return true;
}
std::string quote(const std::string& value) {
    std::string result{"'"};
    for (const char character : value) {
        if (character == '\'') result += "'\\''";
        else result += character;
    }
    return result + "'";
}
}

same_compilation_transform_receipt_v1 deliver_same_compilation_transform_v1(
    const same_compilation_transform_request_v1& request) noexcept {
    same_compilation_transform_receipt_v1 receipt;
    receipt.reflected_field_before = request.reflected_field;
    receipt.reflected_field_after = request.reflected_field;
    receipt.source_file = request.source_file;
    receipt.source_line = request.source_line;
    if (request.prelude_transform_source.empty() || request.ordinary_source.empty()
        || request.host_compiler.empty() || request.compiler_api_identity.empty()
        || request.cache_directory.empty()) {
        receipt.status = same_compilation_transform_status_v1::invalid_request;
        return receipt;
    }
    if (request.requested_generations > request.maximum_generations) {
        receipt.status = same_compilation_transform_status_v1::recursion_limit;
        return receipt;
    }
    try {
        const auto temp = (std::filesystem::path(request.cache_directory) / "early").string();
        build_context context{{request.prelude_transform_source, request.host_compiler,
            request.compiler_api_identity, "", temp}};
        transform_cache_request_v1 cache{{request.prelude_transform_source,
            request.compiler_api_identity, "self-transform-v1", request.host_compiler,
            "native-host", {}, "trusted-in-process"}, request.cache_directory, false,
            build_transform, &context};
        const auto cached = get_or_build_cached_transform_v1(cache);
        if (cached.status != transform_cache_status_v1::success) {
            receipt.status = same_compilation_transform_status_v1::transform_compilation_failed;
            return receipt;
        }
        receipt.cache_hit = cached.warm_hit;
        receipt.transform_artifact = cached.artifact_path;
        void* library = dlopen(cached.artifact_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (library == nullptr) {
            receipt.status = same_compilation_transform_status_v1::transform_load_failed;
            return receipt;
        }
        auto transform = reinterpret_cast<bool (*)(std::uint64_t*) noexcept>(
            dlsym(library, "cellerator_self_transform"));
        const bool transformed = transform != nullptr && transform(&receipt.reflected_field_after);
        dlclose(library);
        if (!transformed) {
            if (!request.allow_clean_fallback) {
                receipt.status = same_compilation_transform_status_v1::transform_failed;
                return receipt;
            }
            receipt.fallback_used = true;
            receipt.reflected_field_after = receipt.reflected_field_before;
        }
        std::filesystem::create_directories(request.cache_directory);
        const auto source = std::filesystem::path(request.cache_directory) / "ordinary.cc";
        const auto object = std::filesystem::path(request.cache_directory) / "ordinary.o";
        std::ofstream output(source);
        output << request.ordinary_source;
        output.close();
        const std::string command = quote(request.host_compiler) + " -std=c++17 -c "
            + quote(source.string()) + " -o " + quote(object.string());
        if (!output || std::system(command.c_str()) != 0) {
            receipt.status = same_compilation_transform_status_v1::object_emission_failed;
            return receipt;
        }
        receipt.ordinary_object = object.string();
    } catch (...) {
        receipt.status = same_compilation_transform_status_v1::object_emission_failed;
    }
    return receipt;
}

}  // namespace cellerator::compiler::pass::v1
