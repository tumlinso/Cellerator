#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace Cellerator::compiler::frontend::source {

enum class generated_identifier_domain_v1 : std::uint8_t { local_symbol = 1, type, module, link_name };
struct generated_identifier_v1 {
    generated_identifier_domain_v1 domain = generated_identifier_domain_v1::local_symbol;
    std::uint64_t content_hash = 0;
    std::string spelling;
    bool emitted_after_preprocessing = true;
};

[[nodiscard]] generated_identifier_v1 make_generated_identifier_v1(
    generated_identifier_domain_v1 domain, std::string_view canonical_content);
[[nodiscard]] bool is_reserved_generated_identifier_v1(std::string_view spelling) noexcept;

} // namespace Cellerator::compiler::frontend::source
