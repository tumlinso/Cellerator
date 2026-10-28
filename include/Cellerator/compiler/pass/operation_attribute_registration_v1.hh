#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace cellerator::compiler::pass::v1 {

enum class extension_entity_kind_v1 : std::uint8_t { operation = 0, type, attribute };
enum extension_protocol_v1 : std::uint32_t {
    extension_text_syntax_v1 = 1U << 0U,
    extension_effects_v1 = 1U << 1U,
    extension_reflection_v1 = 1U << 2U,
    extension_state_transfer_v1 = 1U << 3U,
    extension_verification_v1 = 1U << 4U,
    extension_cost_v1 = 1U << 5U,
    extension_lowering_v1 = 1U << 6U,
};

using extension_protocol_callback_v1 = bool (*)(void*) noexcept;

struct extension_entity_v1 {
    extension_entity_kind_v1 kind = extension_entity_kind_v1::operation;
    std::string local_name;
    std::uint32_t protocols = 0;
    extension_protocol_callback_v1 text_syntax = nullptr;
    extension_protocol_callback_v1 effects = nullptr;
    extension_protocol_callback_v1 reflection = nullptr;
    extension_protocol_callback_v1 state_transfer = nullptr;
    extension_protocol_callback_v1 verification = nullptr;
    extension_protocol_callback_v1 cost = nullptr;
    extension_protocol_callback_v1 lowering = nullptr;
};

struct extension_namespace_v1 {
    std::string name;
    std::uint32_t abi_version = 1;
    std::vector<extension_entity_v1> entities;
};

enum class extension_registration_status_v1 : std::uint8_t {
    success = 0,
    invalid_namespace,
    invalid_entity,
    duplicate_namespace,
};

class extension_registry_v1 {
public:
    [[nodiscard]] extension_registration_status_v1 register_namespace(
        extension_namespace_v1 descriptor);
    [[nodiscard]] const extension_namespace_v1* find_namespace(
        std::string_view name) const noexcept;
    [[nodiscard]] const extension_entity_v1* find_entity(
        std::string_view qualified_name, extension_entity_kind_v1 kind) const noexcept;

private:
    std::vector<extension_namespace_v1> namespaces_;
};

}  // namespace cellerator::compiler::pass::v1
