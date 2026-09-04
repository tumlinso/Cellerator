#include <Cellerator/compiler/ir/semantic/freeze_semantic_ir_module_and_symbol_scopes_v1.hh>

#include <algorithm>
#include <tuple>
#include <unordered_set>
#include <utility>

namespace Cellerator::compiler::ir::semantic {
namespace {

bool valid_child(semantic_scope_kind_v1 parent, semantic_scope_kind_v1 child) noexcept {
    switch (child) {
    case semantic_scope_kind_v1::program:
        return false;
    case semantic_scope_kind_v1::module:
        return parent == semantic_scope_kind_v1::program;
    case semantic_scope_kind_v1::translation_unit:
        return parent == semantic_scope_kind_v1::module;
    case semantic_scope_kind_v1::function:
        return parent == semantic_scope_kind_v1::translation_unit;
    case semantic_scope_kind_v1::named_field:
        return parent == semantic_scope_kind_v1::translation_unit ||
            parent == semantic_scope_kind_v1::function;
    case semantic_scope_kind_v1::anonymous_field:
        return parent == semantic_scope_kind_v1::function ||
            parent == semantic_scope_kind_v1::named_field ||
            parent == semantic_scope_kind_v1::anonymous_field;
    }
    return false;
}

semantic_scope_diagnostic_v1 make_diagnostic(
    semantic_scope_diagnostic_code_v1 code,
    std::string message,
    semantic_scope_id_v1 scope = invalid_semantic_scope_id_v1,
    std::uint64_t symbol = 0) {
    return {code, scope, symbol, std::move(message)};
}

}  // namespace

const semantic_scope_definition_v1* frozen_semantic_scope_module_v1::scope(
    semantic_scope_id_v1 id) const noexcept {
    return id < scopes_.size() ? &scopes_[id] : nullptr;
}

const semantic_symbol_definition_v1* frozen_semantic_scope_module_v1::symbol(
    std::uint64_t stable_identity) const noexcept {
    const auto found = std::lower_bound(
        symbols_.begin(), symbols_.end(), stable_identity,
        [](const semantic_symbol_definition_v1& candidate, std::uint64_t identity) {
            return candidate.stable_identity < identity;
        });
    return found != symbols_.end() && found->stable_identity == stable_identity
        ? &*found : nullptr;
}

semantic_scope_id_v1 frozen_semantic_scope_module_v1::owning_translation_unit(
    semantic_scope_id_v1 id) const noexcept {
    while (id < scopes_.size()) {
        if (scopes_[id].kind == semantic_scope_kind_v1::translation_unit) return id;
        id = scopes_[id].parent;
    }
    return invalid_semantic_scope_id_v1;
}

const semantic_symbol_definition_v1* frozen_semantic_scope_module_v1::resolve_local(
    semantic_scope_id_v1 id, std::string_view name) const noexcept {
    while (id < scopes_.size()) {
        const auto found = std::find_if(symbols_.begin(), symbols_.end(),
            [id, name](const semantic_symbol_definition_v1& candidate) {
                return candidate.owner_scope == id && candidate.name == name;
            });
        if (found != symbols_.end()) return &*found;
        id = scopes_[id].parent;
    }
    return nullptr;
}

const semantic_symbol_definition_v1* frozen_semantic_scope_module_v1::resolve_imported(
    semantic_scope_id_v1 id, std::string_view name) const noexcept {
    while (id < scopes_.size()) {
        for (const auto& imported : imports_) {
            if (imported.importing_scope != id) continue;
            const auto* candidate = symbol(imported.symbol_identity);
            if (candidate != nullptr && candidate->name == name) return candidate;
        }
        id = scopes_[id].parent;
    }
    return nullptr;
}

const std::vector<semantic_scope_definition_v1>&
frozen_semantic_scope_module_v1::scopes() const noexcept {
    return scopes_;
}

const std::vector<semantic_symbol_definition_v1>&
frozen_semantic_scope_module_v1::symbols() const noexcept {
    return symbols_;
}

std::optional<frozen_semantic_scope_module_v1>
freeze_semantic_ir_module_and_symbol_scopes_v1(
    semantic_scope_module_definition_v1 definition,
    semantic_scope_diagnostic_v1* diagnostic) noexcept {
    auto fail = [&](semantic_scope_diagnostic_v1 failure)
        -> std::optional<frozen_semantic_scope_module_v1> {
        if (diagnostic != nullptr) *diagnostic = std::move(failure);
        return std::nullopt;
    };
    if (definition.schema_version != semantic_scope_schema_version_v1) {
        return fail(make_diagnostic(semantic_scope_diagnostic_code_v1::unsupported_schema,
                                    "unsupported semantic scope schema"));
    }
    std::sort(definition.scopes.begin(), definition.scopes.end(),
              [](const auto& left, const auto& right) { return left.id < right.id; });
    if (definition.scopes.empty() ||
        definition.scopes.front().kind != semantic_scope_kind_v1::program) {
        return fail(make_diagnostic(semantic_scope_diagnostic_code_v1::missing_program_scope,
                                    "exactly one program root is required"));
    }
    std::unordered_set<std::uint64_t> scope_identities;
    for (std::size_t index = 0; index < definition.scopes.size(); ++index) {
        const auto& item = definition.scopes[index];
        if (item.id != index || item.stable_identity == 0 || item.name.empty() ||
            !scope_identities.insert(item.stable_identity).second) {
            return fail(make_diagnostic(semantic_scope_diagnostic_code_v1::invalid_scope_identity,
                                        "scope ids must be dense and stable identities unique", item.id));
        }
        if (index == 0) {
            if (item.parent != invalid_semantic_scope_id_v1) {
                return fail(make_diagnostic(semantic_scope_diagnostic_code_v1::invalid_scope_parent,
                                            "program root cannot have a parent", item.id));
            }
            continue;
        }
        if (item.parent >= index) {
            return fail(make_diagnostic(semantic_scope_diagnostic_code_v1::invalid_scope_parent,
                                        "scope parent must precede its child", item.id));
        }
        if (!valid_child(definition.scopes[item.parent].kind, item.kind)) {
            return fail(make_diagnostic(semantic_scope_diagnostic_code_v1::invalid_scope_nesting,
                                        "semantic scope kind is not legal under its parent", item.id));
        }
        const auto duplicate = std::find_if(definition.scopes.begin(),
            definition.scopes.begin() + index, [&item](const auto& prior) {
                return prior.parent == item.parent && prior.kind == item.kind &&
                    prior.name == item.name;
            });
        if (duplicate != definition.scopes.begin() + index) {
            return fail(make_diagnostic(semantic_scope_diagnostic_code_v1::duplicate_scope,
                                        "duplicate named scope under one owner", item.id));
        }
    }

    std::sort(definition.symbols.begin(), definition.symbols.end(),
              [](const auto& left, const auto& right) {
                  return left.stable_identity < right.stable_identity;
              });
    for (std::size_t index = 0; index < definition.symbols.size(); ++index) {
        const auto& item = definition.symbols[index];
        if (item.stable_identity == 0 || item.name.empty() ||
            item.owner_scope >= definition.scopes.size()) {
            return fail(make_diagnostic(semantic_scope_diagnostic_code_v1::invalid_symbol,
                                        "symbol requires identity, name, and valid owner",
                                        item.owner_scope, item.stable_identity));
        }
        if (index != 0 && definition.symbols[index - 1].stable_identity == item.stable_identity) {
            return fail(make_diagnostic(semantic_scope_diagnostic_code_v1::duplicate_symbol,
                                        "duplicate semantic symbol identity",
                                        item.owner_scope, item.stable_identity));
        }
        const auto duplicate = std::find_if(definition.symbols.begin(),
            definition.symbols.begin() + index, [&item](const auto& prior) {
                return prior.owner_scope == item.owner_scope && prior.kind == item.kind &&
                    prior.name == item.name;
            });
        if (duplicate != definition.symbols.begin() + index) {
            return fail(make_diagnostic(semantic_scope_diagnostic_code_v1::duplicate_symbol,
                                        "duplicate semantic symbol in one scope",
                                        item.owner_scope, item.stable_identity));
        }
    }

    frozen_semantic_scope_module_v1 result;
    result.scopes_ = std::move(definition.scopes);
    result.symbols_ = std::move(definition.symbols);
    auto translation_unit = [&result](semantic_scope_id_v1 id) {
        return result.owning_translation_unit(id);
    };
    std::sort(definition.export_authorizations.begin(), definition.export_authorizations.end(),
              [](const auto& left, const auto& right) {
                  return std::tie(left.symbol_identity, left.exporting_translation_unit,
                                  left.importing_translation_unit) <
                      std::tie(right.symbol_identity, right.exporting_translation_unit,
                               right.importing_translation_unit);
              });
    for (std::size_t index = 0; index < definition.export_authorizations.size(); ++index) {
        const auto& item = definition.export_authorizations[index];
        const auto* exported = result.symbol(item.symbol_identity);
        if (exported == nullptr || item.exporting_translation_unit >= result.scopes_.size() ||
            item.importing_translation_unit >= result.scopes_.size() ||
            result.scopes_[item.exporting_translation_unit].kind != semantic_scope_kind_v1::translation_unit ||
            result.scopes_[item.importing_translation_unit].kind != semantic_scope_kind_v1::translation_unit ||
            translation_unit(exported->owner_scope) != item.exporting_translation_unit ||
            item.exporting_translation_unit == item.importing_translation_unit) {
            return fail(make_diagnostic(semantic_scope_diagnostic_code_v1::invalid_export_authorization,
                                        "export must name the defining and a distinct importing translation unit",
                                        item.importing_translation_unit, item.symbol_identity));
        }
        if (index != 0 && definition.export_authorizations[index - 1].symbol_identity == item.symbol_identity &&
            definition.export_authorizations[index - 1].exporting_translation_unit == item.exporting_translation_unit &&
            definition.export_authorizations[index - 1].importing_translation_unit == item.importing_translation_unit) {
            return fail(make_diagnostic(semantic_scope_diagnostic_code_v1::duplicate_export_authorization,
                                        "duplicate cross-translation-unit export authorization",
                                        item.importing_translation_unit, item.symbol_identity));
        }
    }
    result.export_authorizations_ = std::move(definition.export_authorizations);

    std::sort(definition.imports.begin(), definition.imports.end(),
              [](const auto& left, const auto& right) {
                  return std::tie(left.importing_scope, left.symbol_identity) <
                      std::tie(right.importing_scope, right.symbol_identity);
              });
    for (std::size_t index = 0; index < definition.imports.size(); ++index) {
        const auto& item = definition.imports[index];
        const auto* imported = result.symbol(item.symbol_identity);
        const auto importer = translation_unit(item.importing_scope);
        const auto exporter = imported == nullptr
            ? invalid_semantic_scope_id_v1 : translation_unit(imported->owner_scope);
        const bool authorized = std::any_of(result.export_authorizations_.begin(),
            result.export_authorizations_.end(), [&](const auto& authorization) {
                return authorization.symbol_identity == item.symbol_identity &&
                    authorization.exporting_translation_unit == exporter &&
                    authorization.importing_translation_unit == importer;
            });
        if (imported == nullptr || importer == invalid_semantic_scope_id_v1 || !authorized) {
            return fail(make_diagnostic(semantic_scope_diagnostic_code_v1::unauthorized_import,
                                        "import lacks exact cross-translation-unit authorization",
                                        item.importing_scope, item.symbol_identity));
        }
        if (index != 0 && definition.imports[index - 1].importing_scope == item.importing_scope &&
            definition.imports[index - 1].symbol_identity == item.symbol_identity) {
            return fail(make_diagnostic(semantic_scope_diagnostic_code_v1::duplicate_import,
                                        "duplicate imported semantic symbol",
                                        item.importing_scope, item.symbol_identity));
        }
    }
    result.imports_ = std::move(definition.imports);
    if (diagnostic != nullptr) *diagnostic = {};
    return result;
}

}  // namespace Cellerator::compiler::ir::semantic
