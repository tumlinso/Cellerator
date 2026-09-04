#include <Cellerator/compiler/pass/cache_compiled_transforms_safely_v1.hh>

#include <chrono>
#include <filesystem>

namespace cellerator::compiler::pass::v1 {
namespace {
void mix(std::uint64_t& hash, const std::string& text) {
    for (const unsigned char value : text) {
        hash ^= value;
        hash *= 1099511628211ULL;
    }
    hash ^= 0xffU;
    hash *= 1099511628211ULL;
}
}

std::uint64_t transform_cache_key_v1(
    const transform_cache_identity_v1& identity) noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    mix(hash, identity.source_identity);
    mix(hash, identity.compiler_api_identity);
    mix(hash, identity.extension_abi_identity);
    mix(hash, identity.toolchain_identity);
    mix(hash, identity.target_host_identity);
    for (const auto& dependency : identity.dependency_identities) mix(hash, dependency);
    mix(hash, identity.trust_policy_identity);
    return hash;
}

transform_cache_receipt_v1 get_or_build_cached_transform_v1(
    const transform_cache_request_v1& request) noexcept {
    const auto start = std::chrono::steady_clock::now();
    transform_cache_receipt_v1 receipt;
    receipt.identity_key = transform_cache_key_v1(request.identity);
    if (request.cache_directory.empty() || request.build == nullptr
        || request.identity.source_identity.empty()
        || request.identity.compiler_api_identity.empty()
        || request.identity.extension_abi_identity.empty()
        || request.identity.toolchain_identity.empty()
        || request.identity.target_host_identity.empty()
        || request.identity.trust_policy_identity.empty()) {
        receipt.status = transform_cache_status_v1::invalid_request;
        return receipt;
    }
    try {
        std::filesystem::create_directories(request.cache_directory);
        const auto base = std::filesystem::path(request.cache_directory)
            / std::to_string(receipt.identity_key);
        const auto artifact = base.string() + ".so";
        const auto temporary = base.string() + ".tmp";
        receipt.artifact_path = artifact;
        if (std::filesystem::is_regular_file(artifact)) {
            receipt.warm_hit = true;
        } else {
            receipt.temporary_path = temporary;
            if (!request.build(temporary, request.user_data)) {
                receipt.status = transform_cache_status_v1::build_failed;
            } else {
                if (request.keep_temps) {
                    std::filesystem::copy_file(temporary, artifact,
                        std::filesystem::copy_options::overwrite_existing);
                } else {
                    std::filesystem::rename(temporary, artifact);
                    receipt.temporary_path.clear();
                }
            }
        }
    } catch (...) {
        receipt.status = transform_cache_status_v1::publish_failed;
    }
    receipt.elapsed_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count());
    return receipt;
}

}  // namespace cellerator::compiler::pass::v1
