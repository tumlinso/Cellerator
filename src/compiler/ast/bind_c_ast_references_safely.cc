#include <Cellerator/compiler/ast/bind_c_ast_references_safely_v1.hh>

#include <algorithm>
#include <tuple>
#include <utility>

namespace Cellerator::compiler::ast {
namespace {

auto order_key(const cxx_ast_reference_key_v1& key) noexcept {
    return std::tie(key.kind, key.semantic_owner, key.canonical_source,
                    key.declaration_identity, key.type_identity);
}

void set_error(std::string* error, std::string message) {
    if (error != nullptr) {
        *error = std::move(message);
    }
}

} // namespace

std::uint64_t cxx_ast_reference_table_v1::registry_identity() const noexcept {
    return registry_identity_;
}

std::uint32_t cxx_ast_reference_table_v1::generation() const noexcept {
    return generation_;
}

std::uint32_t cxx_ast_reference_table_v1::adapter_version() const noexcept {
    return adapter_version_;
}

std::size_t cxx_ast_reference_table_v1::size() const noexcept {
    return keys_.size();
}

std::optional<cxx_ast_reference_v1>
cxx_ast_reference_table_v1::reference(const cxx_ast_reference_key_v1& key) const noexcept {
    const auto found = std::lower_bound(keys_.begin(), keys_.end(), key,
                                        [](const auto& left, const auto& right) {
                                            return order_key(left) < order_key(right);
                                        });
    if (found == keys_.end() || !(*found == key)) {
        return std::nullopt;
    }
    return cxx_ast_reference_v1{registry_identity_, generation_, adapter_version_,
                                static_cast<std::uint32_t>(found - keys_.begin()), key.kind, {}};
}

const cxx_ast_reference_key_v1*
cxx_ast_reference_table_v1::resolve(cxx_ast_reference_v1 reference) const noexcept {
    if (!reference.valid() || reference.registry_identity != registry_identity_ ||
        reference.generation != generation_ || reference.adapter_version != adapter_version_ ||
        reference.slot >= keys_.size() || keys_[reference.slot].kind != reference.kind) {
        return nullptr;
    }
    return &keys_[reference.slot];
}

std::optional<cxx_ast_reference_table_v1>
freeze_cxx_ast_references_v1(std::uint64_t registry_identity,
                             std::uint32_t generation,
                             std::uint32_t adapter_version,
                             std::vector<cxx_ast_reference_key_v1> keys,
                             std::string* error) {
    if (registry_identity == 0 || generation == 0 || adapter_version == 0) {
        set_error(error, "C++ AST reference table identities must be nonzero");
        return std::nullopt;
    }
    for (const auto& key : keys) {
        if (key.semantic_owner == 0 || key.canonical_source == 0 ||
            key.declaration_identity == 0) {
            set_error(error, "C++ AST reference key is incomplete");
            return std::nullopt;
        }
    }
    std::sort(keys.begin(), keys.end(), [](const auto& left, const auto& right) {
        return order_key(left) < order_key(right);
    });
    if (std::adjacent_find(keys.begin(), keys.end()) != keys.end()) {
        set_error(error, "duplicate C++ AST reference key");
        return std::nullopt;
    }
    cxx_ast_reference_table_v1 table;
    table.registry_identity_ = registry_identity;
    table.generation_ = generation;
    table.adapter_version_ = adapter_version;
    table.keys_ = std::move(keys);
    return table;
}

std::vector<cxx_ast_reference_rebuild_v1>
rebuild_cxx_ast_references_v1(const cxx_ast_reference_table_v1& old_table,
                              const cxx_ast_reference_table_v1& new_table) {
    std::vector<cxx_ast_reference_rebuild_v1> result;
    result.reserve(old_table.size());
    for (std::size_t slot = 0; slot < old_table.size(); ++slot) {
        const cxx_ast_reference_v1 old_reference{
            old_table.registry_identity(), old_table.generation(), old_table.adapter_version(),
            static_cast<std::uint32_t>(slot),
            cxx_ast_entity_kind_v1::declaration, {}};
        auto typed_reference = old_reference;
        for (const auto kind : {cxx_ast_entity_kind_v1::declaration,
                                cxx_ast_entity_kind_v1::expression,
                                cxx_ast_entity_kind_v1::type,
                                cxx_ast_entity_kind_v1::template_entity,
                                cxx_ast_entity_kind_v1::constant}) {
            typed_reference.kind = kind;
            if (const auto* key = old_table.resolve(typed_reference)) {
                result.push_back({typed_reference, new_table.reference(*key)});
                break;
            }
        }
    }
    return result;
}

} // namespace Cellerator::compiler::ast
