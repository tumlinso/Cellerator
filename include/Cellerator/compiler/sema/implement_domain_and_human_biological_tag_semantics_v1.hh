#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>

namespace cellerator::compiler::sema::v1 {

struct domain_identity {
    std::uint64_t low = 0;
    std::uint64_t high = 0;
};

using biological_tag_id = std::uint32_t;
inline constexpr biological_tag_id no_biological_tag = 0u;

struct domain_type {
    domain_identity identity{};
    biological_tag_id diagnostic_tag = no_biological_tag;
    bool domain_erased = false;
};

class biological_tag_registry {
public:
    biological_tag_registry();

    biological_tag_id register_tag(std::string spelling);
    biological_tag_id find_tag(std::string_view spelling) const noexcept;
    std::string_view spelling(biological_tag_id tag) const noexcept;

private:
    std::unordered_map<std::string, biological_tag_id> ids_;
    std::unordered_map<biological_tag_id, std::string> spellings_;
    biological_tag_id next_id_ = 1u;
};

bool same_nominal_domain(const domain_type &left,
                         const domain_type &right) noexcept;
domain_type erase_domain_tag(domain_type domain) noexcept;
domain_type erase_nominal_domain(domain_type domain) noexcept;
bool can_explicitly_cast_domain(const domain_type &source,
                                const domain_type &destination,
                                bool unsafe_authorized) noexcept;

}  // namespace cellerator::compiler::sema::v1
