#include <Cellerator/compiler/frontend/source/cache_preprocessed_activated_headers_safely_v1.hh>

#include <functional>
#include <utility>

namespace Cellerator::compiler::frontend::source {

std::size_t activated_header_cache_key_hash_v1::operator()(const activated_header_cache_key_v1& key) const noexcept {
    std::size_t hash = 0xcbf29ce484222325ULL;
    for (const auto* component : {&key.file_content, &key.pragma_revision, &key.macro_environment,
                                  &key.include_context, &key.frontend_adapter_identity}) {
        hash ^= std::hash<std::string>{}(*component) + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
    }
    return hash;
}

void activated_header_cache_v1::store(activated_header_cache_key_v1 key, std::string product) {
    products_.insert_or_assign(std::move(key), std::move(product));
}

std::optional<std::string_view> activated_header_cache_v1::find(const activated_header_cache_key_v1& key) const noexcept {
    const auto found = products_.find(key);
    return found == products_.end() ? std::nullopt : std::optional<std::string_view>(found->second);
}

} // namespace Cellerator::compiler::frontend::source
