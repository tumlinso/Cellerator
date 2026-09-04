#include <Cellerator/compiler/ast/implement_cellerator_symbol_tables_and_scopes_v1.hh>

#include <algorithm>
#include <unordered_set>

namespace Cellerator::compiler::ast {
namespace {

bool overloadable(symbol_kind_v1 kind) noexcept {
    return kind == symbol_kind_v1::field || kind == symbol_kind_v1::candidate ||
           kind == symbol_kind_v1::compiler_pass || kind == symbol_kind_v1::imported_program;
}

bool matches(const symbol_declaration_v1& declaration,
             const symbol_lookup_request_v1& request) noexcept {
    return declaration.name == request.name &&
           (!request.kind || declaration.kind == *request.kind) &&
           (!request.signature_identity ||
            declaration.signature_identity == *request.signature_identity);
}

symbol_lookup_result_v1 classify(std::vector<const symbol_declaration_v1*> candidates,
                                 symbol_scope_id_v1 declaring_scope) {
    if (candidates.empty()) {
        return {symbol_lookup_status_v1::not_found, {}, invalid_symbol_scope_v1};
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto* left, const auto* right) {
        return left->identity < right->identity;
    });
    if (candidates.size() == 1) {
        return {symbol_lookup_status_v1::resolved, std::move(candidates), declaring_scope};
    }
    const auto kind = candidates.front()->kind;
    const bool same_kind = std::all_of(candidates.begin(), candidates.end(), [kind](const auto* item) {
        return item->kind == kind;
    });
    const bool distinct_signatures = std::adjacent_find(
        candidates.begin(), candidates.end(), [](const auto* left, const auto* right) {
            return left->signature_identity == right->signature_identity;
        }) == candidates.end();
    const auto status = same_kind && overloadable(kind) && distinct_signatures
                            ? symbol_lookup_status_v1::overload_set
                            : symbol_lookup_status_v1::ambiguous;
    return {status, std::move(candidates), declaring_scope};
}

} // namespace

const symbol_scope_v1* symbol_table_v1::scope(symbol_scope_id_v1 id) const noexcept {
    return id < scopes_.size() ? &scopes_[id] : nullptr;
}

std::size_t symbol_table_v1::scope_count() const noexcept { return scopes_.size(); }

symbol_lookup_result_v1 symbol_table_v1::lookup(symbol_lookup_request_v1 request) const {
    if (request.name.empty() || !scope(request.scope)) {
        return {};
    }
    auto current = request.scope;
    while (current != invalid_symbol_scope_v1) {
        const auto& lexical = scopes_[current];
        std::vector<const symbol_declaration_v1*> local;
        for (const auto& declaration : lexical.declarations) {
            if (matches(declaration, request)) local.push_back(&declaration);
        }
        if (!local.empty()) return classify(std::move(local), current);
        if (request.qualified) break;

        std::vector<const symbol_declaration_v1*> imported;
        for (const auto imported_scope : lexical.imports) {
            for (const auto& declaration : scopes_[imported_scope].declarations) {
                if (matches(declaration, request)) imported.push_back(&declaration);
            }
        }
        if (!imported.empty()) return classify(std::move(imported), invalid_symbol_scope_v1);
        current = lexical.parent;
    }
    return {symbol_lookup_status_v1::not_found, {}, invalid_symbol_scope_v1};
}

std::optional<symbol_table_v1>
freeze_symbol_table_v1(std::vector<symbol_scope_v1> scopes, std::string* error) {
    auto fail = [&](std::string message) -> std::optional<symbol_table_v1> {
        if (error) *error = std::move(message);
        return std::nullopt;
    };
    if (scopes.empty()) return fail("symbol table requires a root scope");
    std::sort(scopes.begin(), scopes.end(), [](const auto& left, const auto& right) {
        return left.id < right.id;
    });
    for (std::size_t index = 0; index < scopes.size(); ++index) {
        auto& item = scopes[index];
        if (item.id != index) return fail("scope ids must be dense and rooted at zero");
        if (index == 0 && item.parent != invalid_symbol_scope_v1)
            return fail("root scope cannot have a parent");
        if (index != 0 && (item.parent >= index || item.parent == invalid_symbol_scope_v1))
            return fail("scope parent must precede its child");
        std::sort(item.imports.begin(), item.imports.end());
        if (std::adjacent_find(item.imports.begin(), item.imports.end()) != item.imports.end())
            return fail("duplicate imported scope");
        for (const auto imported : item.imports)
            if (imported >= scopes.size() || imported == item.id)
                return fail("invalid imported scope");
        std::sort(item.declarations.begin(), item.declarations.end(), [](const auto& left, const auto& right) {
            if (left.name != right.name) return left.name < right.name;
            if (left.kind != right.kind) return left.kind < right.kind;
            if (left.signature_identity != right.signature_identity)
                return left.signature_identity < right.signature_identity;
            return left.identity < right.identity;
        });
        std::unordered_set<std::uint64_t> identities;
        for (const auto& declaration : item.declarations) {
            if (declaration.identity == 0 || declaration.name.empty() ||
                !identities.insert(declaration.identity).second)
                return fail("invalid or duplicate declaration identity");
        }
        for (std::size_t declaration = 1; declaration < item.declarations.size(); ++declaration) {
            const auto& left = item.declarations[declaration - 1];
            const auto& right = item.declarations[declaration];
            if (left.name == right.name && left.kind == right.kind &&
                left.signature_identity == right.signature_identity)
                return fail("duplicate declaration in one scope");
        }
    }
    symbol_table_v1 table;
    table.scopes_ = std::move(scopes);
    if (error) error->clear();
    return table;
}

} // namespace Cellerator::compiler::ast
