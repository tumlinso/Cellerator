#include <Cellerator/compiler/pass/define_extensible_operation_type_and_attribute_registrat_v1.hh>

#include <utility>

namespace cellerator::compiler::pass::v1 {
namespace {
bool valid_name(std::string_view name) {
    return !name.empty() && name.find('.') == std::string_view::npos;
}
bool protocols_match(const extension_entity_v1& entity) {
    const extension_protocol_callback_v1 callbacks[]{entity.text_syntax, entity.effects,
        entity.reflection, entity.state_transfer, entity.verification, entity.cost,
        entity.lowering};
    for (std::uint32_t index = 0; index < 7; ++index) {
        if (((entity.protocols >> index) & 1U) != (callbacks[index] != nullptr)) {
            return false;
        }
    }
    return true;
}
}

extension_registration_status_v1 extension_registry_v1::register_namespace(
    extension_namespace_v1 descriptor) {
    if (!valid_name(descriptor.name) || descriptor.abi_version == 0) {
        return extension_registration_status_v1::invalid_namespace;
    }
    if (find_namespace(descriptor.name) != nullptr) {
        return extension_registration_status_v1::duplicate_namespace;
    }
    for (std::size_t index = 0; index < descriptor.entities.size(); ++index) {
        const auto& entity = descriptor.entities[index];
        if (!valid_name(entity.local_name) || !protocols_match(entity)) {
            return extension_registration_status_v1::invalid_entity;
        }
        for (std::size_t prior = 0; prior < index; ++prior) {
            if (descriptor.entities[prior].kind == entity.kind
                && descriptor.entities[prior].local_name == entity.local_name) {
                return extension_registration_status_v1::invalid_entity;
            }
        }
    }
    namespaces_.push_back(std::move(descriptor));
    return extension_registration_status_v1::success;
}

const extension_namespace_v1* extension_registry_v1::find_namespace(
    std::string_view name) const noexcept {
    for (const auto& descriptor : namespaces_) {
        if (descriptor.name == name) return &descriptor;
    }
    return nullptr;
}

const extension_entity_v1* extension_registry_v1::find_entity(
    std::string_view qualified_name, extension_entity_kind_v1 kind) const noexcept {
    const auto dot = qualified_name.find('.');
    if (dot == std::string_view::npos) return nullptr;
    const auto* descriptor = find_namespace(qualified_name.substr(0, dot));
    if (descriptor == nullptr) return nullptr;
    const auto local = qualified_name.substr(dot + 1);
    for (const auto& entity : descriptor->entities) {
        if (entity.kind == kind && entity.local_name == local) return &entity;
    }
    return nullptr;
}

}  // namespace cellerator::compiler::pass::v1
