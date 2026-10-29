#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::planning {

enum class custom_candidate_origin_v1 : std::uint8_t {
    source = 1u,
    inline_ir,
    external_library,
    migrated_provider,
};

enum custom_candidate_protocol_v1 : std::uint32_t {
    custom_protocol_prepare_v1 = 1u << 0u,
    custom_protocol_execute_v1 = 1u << 1u,
    custom_protocol_estimate_v1 = 1u << 2u,
    custom_protocol_profile_v1 = 1u << 3u,
    custom_protocol_reflect_v1 = 1u << 4u,
};

enum class missing_protocol_behavior_v1 : std::uint8_t {
    reject = 0u,
    opaque_passthrough = 1u,
};

struct custom_candidate_registration_v1 {
    std::uint64_t candidate_identity = 0u;
    std::uint64_t provider_identity = 0u;
    std::uint64_t operation_identity = 0u;
    custom_candidate_origin_v1 origin = custom_candidate_origin_v1::source;
    std::uint32_t provided_protocols = 0u;
    std::uint32_t required_protocols = 0u;
    missing_protocol_behavior_v1 missing_behavior =
        missing_protocol_behavior_v1::reject;
    std::string stable_name;
    std::string source_locator;
    std::string entry_symbol;
    std::vector<std::uint8_t> opaque_payload;
};

enum class custom_candidate_registration_code_v1 : std::uint8_t {
    ok = 0u,
    invalid_identity,
    invalid_name,
    invalid_origin,
    missing_source_locator,
    missing_entry_symbol,
    incomplete_protocol,
    duplicate_candidate,
    malformed_ir,
};

struct custom_candidate_registry_v1 {
    std::vector<custom_candidate_registration_v1> candidates;
};

[[nodiscard]] custom_candidate_registration_code_v1 register_custom_candidate_v1(
    custom_candidate_registry_v1* registry,
    custom_candidate_registration_v1 candidate);

[[nodiscard]] std::vector<std::uint8_t> write_custom_candidate_binary_ir_v1(
    const custom_candidate_registry_v1& registry);

[[nodiscard]] custom_candidate_registration_code_v1
read_custom_candidate_binary_ir_v1(
    const std::vector<std::uint8_t>& bytes,
    custom_candidate_registry_v1* registry);

[[nodiscard]] std::string write_custom_candidate_text_ir_v1(
    const custom_candidate_registry_v1& registry);

[[nodiscard]] custom_candidate_registration_code_v1
read_custom_candidate_text_ir_v1(
    const std::string& text,
    custom_candidate_registry_v1* registry);

[[nodiscard]] bool equivalent_custom_candidate_registries_v1(
    const custom_candidate_registry_v1& lhs,
    const custom_candidate_registry_v1& rhs) noexcept;

}  // namespace Cellerator::compiler::planning
