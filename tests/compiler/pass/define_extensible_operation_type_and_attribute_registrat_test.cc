#include <Cellerator/compiler/pass/define_extensible_operation_type_and_attribute_registrat_v1.hh>

#include <cassert>

namespace cp = cellerator::compiler::pass::v1;

namespace {
bool protocol(void*) noexcept { return true; }
}

int main() {
    cp::extension_registry_v1 registry;
    cp::extension_namespace_v1 partial{"biology", 1, {
        {cp::extension_entity_kind_v1::operation, "regulate",
            cp::extension_verification_v1 | cp::extension_lowering_v1,
            nullptr, nullptr, nullptr, nullptr, protocol, nullptr, protocol},
        {cp::extension_entity_kind_v1::type, "gene_set",
            cp::extension_text_syntax_v1, protocol},
        {cp::extension_entity_kind_v1::attribute, "confidence", 0},
    }};
    assert(registry.register_namespace(std::move(partial))
        == cp::extension_registration_status_v1::success);
    const auto* operation = registry.find_entity(
        "biology.regulate", cp::extension_entity_kind_v1::operation);
    assert(operation != nullptr);
    assert(operation->cost == nullptr && operation->lowering != nullptr);
    assert(registry.find_entity("biology.unknown",
        cp::extension_entity_kind_v1::operation) == nullptr);
    assert(registry.register_namespace({"biology", 1, {}})
        == cp::extension_registration_status_v1::duplicate_namespace);
}
