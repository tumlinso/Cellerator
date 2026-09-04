#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace cellerator::compiler::ir {

enum class interned_kind : std::uint8_t { type, attribute, opaque_extension };
struct interned_handle { std::uint32_t slot{}; interned_kind kind{}; };

struct intern_result {
    interned_handle handle{};
    bool inserted{};
    bool identity_conflict{};
};

class type_attribute_interner {
public:
    intern_result intern(interned_kind kind, std::string_view canonical_content,
        std::optional<std::uint64_t> asserted_identity = std::nullopt);
    std::string_view content(interned_handle handle) const noexcept;
    std::optional<std::uint64_t> asserted_identity(interned_handle handle) const noexcept;
    std::string serialize(interned_handle handle) const;
    std::size_t size() const noexcept { return records_.size(); }
private:
    struct record {
        interned_kind kind{};
        std::string content{};
        std::optional<std::uint64_t> identity{};
    };
    std::vector<record> records_{};
    std::unordered_map<std::string, std::uint32_t> canonical_{};
    std::unordered_map<std::uint64_t, std::uint32_t> asserted_{};
};

} // namespace cellerator::compiler::ir
