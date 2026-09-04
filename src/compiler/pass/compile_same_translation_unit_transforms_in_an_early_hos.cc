#include <Cellerator/compiler/pass/compile_same_translation_unit_transforms_in_an_early_hos_v1.hh>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace cellerator::compiler::pass::v1 {
namespace {
void hash_text(std::uint64_t& hash, const std::string& text) {
    for (const unsigned char value : text) {
        hash ^= value;
        hash *= 1099511628211ULL;
    }
    hash ^= 0xffU;
    hash *= 1099511628211ULL;
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

std::uint64_t early_host_transform_key_v1(
    const early_host_transform_request_v1& request) noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    hash_text(hash, request.transform_source);
    hash_text(hash, request.host_compiler);
    hash_text(hash, request.compiler_api_identity);
    return hash;
}

early_host_transform_receipt_v1 compile_early_host_transform_v1(
    const early_host_transform_request_v1& request) noexcept {
    early_host_transform_receipt_v1 receipt;
    receipt.cache_key = early_host_transform_key_v1(request);
    if (request.transform_source.empty() || request.host_compiler.empty()
        || request.compiler_api_identity.empty() || request.temporary_directory.empty()) {
        receipt.status = early_host_transform_status_v1::invalid_request;
        return receipt;
    }
    try {
        std::filesystem::create_directories(request.temporary_directory);
        const auto stem = std::to_string(receipt.cache_key);
        const auto source = std::filesystem::path(request.temporary_directory) / (stem + ".cc");
        const auto artifact = std::filesystem::path(request.temporary_directory) / (stem + ".so");
        std::ofstream output(source);
        output << request.transform_source;
        output.close();
        if (!output) {
            receipt.status = early_host_transform_status_v1::source_write_failed;
            return receipt;
        }
        std::ostringstream command;
        command << quote(request.host_compiler) << " -std=c++17 -shared -fPIC ";
        if (!request.include_directory.empty()) {
            command << "-I" << quote(request.include_directory) << ' ';
        }
        command << quote(source.string()) << " -o " << quote(artifact.string());
        if (std::system(command.str().c_str()) != 0) {
            receipt.status = early_host_transform_status_v1::compilation_failed;
            receipt.diagnostic = command.str();
            return receipt;
        }
        receipt.artifact_path = artifact.string();
    } catch (...) {
        receipt.status = early_host_transform_status_v1::source_write_failed;
    }
    return receipt;
}

}  // namespace cellerator::compiler::pass::v1
