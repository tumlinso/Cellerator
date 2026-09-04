#include <Cellerator/compiler/ir/realization/freeze_realization_ir_module_and_target_scopes_v1.hh>

#include <set>
#include <tuple>

namespace cellerator::compiler::ir::realization::v1 {
namespace {

using identity_key_v1 = std::tuple<std::uint64_t, std::uint64_t>;

identity_key_v1 key(stable_identity_v1 identity) noexcept {
    return {identity.high, identity.low};
}

realization_module_status_v1 fail(
    realization_module_status_v1 status,
    std::string* error,
    const char* message) noexcept {
    if (error != nullptr) {
        *error = message;
    }
    return status;
}

bool equal_lineage(
    const realization_lineage_v1& lhs,
    const realization_lineage_v1& rhs) noexcept {
    return lhs.source_identity == rhs.source_identity &&
        lhs.semantic_identity == rhs.semantic_identity &&
        lhs.planning_identity == rhs.planning_identity;
}

} // namespace

realization_module_status_v1 validate_realization_module_v1(
    const realization_module_v1& module,
    std::string* error) noexcept {
    if (module.contract_version != realization_ir_contract_version_v1) {
        return fail(realization_module_status_v1::unsupported_version, error,
            "unsupported realization IR contract version");
    }
    if (!valid(module.identity)) {
        return fail(realization_module_status_v1::missing_identity, error,
            "module identity is required");
    }
    if (module.name.empty()) {
        return fail(realization_module_status_v1::missing_name, error,
            "module name is required");
    }
    if (module.targets.empty()) {
        return fail(realization_module_status_v1::missing_target, error,
            "at least one target scope is required");
    }

    std::set<identity_key_v1> targets;
    for (const auto& target : module.targets) {
        if (!valid(target.identity) || target.target_name.empty() ||
            target.profile_variant.empty()) {
            return fail(realization_module_status_v1::missing_target, error,
                "target identity, name, and profile variant are required");
        }
        if (!targets.insert(key(target.identity)).second) {
            return fail(realization_module_status_v1::duplicate_target, error,
                "target identities must be unique");
        }
    }

    std::set<identity_key_v1> objects;
    for (const auto& object : module.objects) {
        if (!valid(object.identity) || object.name.empty()) {
            return fail(realization_module_status_v1::missing_identity, error,
                "object identity and name are required");
        }
        if (!objects.insert(key(object.identity)).second) {
            return fail(realization_module_status_v1::duplicate_object, error,
                "object identities must be unique");
        }
        if (targets.count(key(object.target_scope)) == 0u) {
            return fail(realization_module_status_v1::unknown_target, error,
                "object target scope is not declared by the module");
        }
        if (!valid(object.lineage.source_identity) ||
            !valid(object.lineage.semantic_identity) ||
            !valid(object.lineage.planning_identity)) {
            return fail(realization_module_status_v1::missing_lineage, error,
                "source, semantic, and planning lineage are required");
        }
    }
    if (error != nullptr) {
        error->clear();
    }
    return realization_module_status_v1::valid;
}

bool equivalent_realization_module_v1(
    const realization_module_v1& lhs,
    const realization_module_v1& rhs) noexcept {
    if (lhs.contract_version != rhs.contract_version ||
        !(lhs.identity == rhs.identity) || lhs.name != rhs.name ||
        lhs.targets.size() != rhs.targets.size() ||
        lhs.objects.size() != rhs.objects.size()) {
        return false;
    }
    for (std::size_t index = 0; index < lhs.targets.size(); ++index) {
        const auto& a = lhs.targets[index];
        const auto& b = rhs.targets[index];
        if (!(a.identity == b.identity) || a.target_name != b.target_name ||
            a.profile_variant != b.profile_variant) {
            return false;
        }
    }
    for (std::size_t index = 0; index < lhs.objects.size(); ++index) {
        const auto& a = lhs.objects[index];
        const auto& b = rhs.objects[index];
        if (!(a.identity == b.identity) || !(a.target_scope == b.target_scope) ||
            a.kind != b.kind || a.name != b.name ||
            !equal_lineage(a.lineage, b.lineage)) {
            return false;
        }
    }
    return true;
}

} // namespace cellerator::compiler::ir::realization::v1
