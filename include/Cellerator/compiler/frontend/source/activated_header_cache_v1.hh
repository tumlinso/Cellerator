#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace Cellerator::compiler::frontend::source {

struct activated_header_cache_key_v1 {
    std::string file_content;
    std::string pragma_revision;
    std::string macro_environment;
    std::string include_context;
    std::string frontend_adapter_identity;
    friend bool operator==(const activated_header_cache_key_v1& a,
                           const activated_header_cache_key_v1& b) noexcept {
        return a.file_content == b.file_content && a.pragma_revision == b.pragma_revision &&
               a.macro_environment == b.macro_environment && a.include_context == b.include_context &&
               a.frontend_adapter_identity == b.frontend_adapter_identity;
    }
};
struct activated_header_cache_key_hash_v1 {
    std::size_t operator()(const activated_header_cache_key_v1& key) const noexcept;
};

class activated_header_cache_v1 {
  public:
    void store(activated_header_cache_key_v1 key, std::string product);
    [[nodiscard]] std::optional<std::string_view> find(const activated_header_cache_key_v1& key) const noexcept;
  private:
    std::unordered_map<activated_header_cache_key_v1, std::string,
                       activated_header_cache_key_hash_v1> products_;
};

} // namespace Cellerator::compiler::frontend::source
