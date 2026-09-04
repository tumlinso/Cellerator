#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace Cellerator::compiler::ast {

enum class cxx_ast_entity_kind_v1 : std::uint8_t {
    declaration = 1,
    expression,
    type,
    template_entity,
    constant,
};

struct cxx_ast_reference_key_v1 {
    cxx_ast_entity_kind_v1 kind = cxx_ast_entity_kind_v1::declaration;
    std::uint64_t semantic_owner = 0;
    std::uint64_t canonical_source = 0;
    std::uint64_t declaration_identity = 0;
    std::uint64_t type_identity = 0;
};

struct cxx_ast_reference_v1 {
    std::uint64_t registry_identity = 0;
    std::uint32_t generation = 0;
    std::uint32_t adapter_version = 0;
    std::uint32_t slot = UINT32_MAX;
    cxx_ast_entity_kind_v1 kind = cxx_ast_entity_kind_v1::declaration;
    std::uint8_t reserved[3]{};

    [[nodiscard]] constexpr bool valid() const noexcept {
        return registry_identity != 0 && generation != 0 && adapter_version != 0 &&
               slot != UINT32_MAX;
    }
};

[[nodiscard]] constexpr bool operator==(const cxx_ast_reference_key_v1& left,
                                        const cxx_ast_reference_key_v1& right) noexcept {
    return left.kind == right.kind && left.semantic_owner == right.semantic_owner &&
           left.canonical_source == right.canonical_source &&
           left.declaration_identity == right.declaration_identity &&
           left.type_identity == right.type_identity;
}

class cxx_ast_reference_table_v1 {
public:
    [[nodiscard]] std::uint64_t registry_identity() const noexcept;
    [[nodiscard]] std::uint32_t generation() const noexcept;
    [[nodiscard]] std::uint32_t adapter_version() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::optional<cxx_ast_reference_v1>
    reference(const cxx_ast_reference_key_v1& key) const noexcept;
    [[nodiscard]] const cxx_ast_reference_key_v1*
    resolve(cxx_ast_reference_v1 reference) const noexcept;

private:
    std::uint64_t registry_identity_ = 0;
    std::uint32_t generation_ = 0;
    std::uint32_t adapter_version_ = 0;
    std::vector<cxx_ast_reference_key_v1> keys_;
    friend std::optional<cxx_ast_reference_table_v1>
    freeze_cxx_ast_references_v1(std::uint64_t, std::uint32_t, std::uint32_t,
                                 std::vector<cxx_ast_reference_key_v1>, std::string*);
};

[[nodiscard]] std::optional<cxx_ast_reference_table_v1>
freeze_cxx_ast_references_v1(std::uint64_t registry_identity,
                             std::uint32_t generation,
                             std::uint32_t adapter_version,
                             std::vector<cxx_ast_reference_key_v1> keys,
                             std::string* error = nullptr);

struct cxx_ast_reference_rebuild_v1 {
    cxx_ast_reference_v1 old_reference{};
    std::optional<cxx_ast_reference_v1> replacement;
};

[[nodiscard]] std::vector<cxx_ast_reference_rebuild_v1>
rebuild_cxx_ast_references_v1(const cxx_ast_reference_table_v1& old_table,
                              const cxx_ast_reference_table_v1& new_table);

} // namespace Cellerator::compiler::ast
