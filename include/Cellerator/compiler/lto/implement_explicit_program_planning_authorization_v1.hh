#pragma once

#include <Cellerator/compiler/lto/freeze_the_ceir_companion_object_artifact_contract_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compiler::lto::v1 {

enum class program_planning_authority_v1 : std::uint8_t {
    none = 0,
    exported_or_named_fields,
    source_policy,
    driver_lto_flag
};

struct program_planning_authorization_v1 {
    program_planning_authority_v1 authority = program_planning_authority_v1::none;
    std::vector<artifact_identity_v1> authorized_fields;
    std::vector<std::string> authorized_sources;
    bool allow_all_ceir_fields = false;
};

struct cross_tu_planning_request_v1 {
    artifact_identity_v1 producer_field{};
    artifact_identity_v1 consumer_field{};
    std::string producer_source;
    std::string consumer_source;
    bool producer_is_exported_or_named = false;
    bool consumer_is_exported_or_named = false;
    bool producer_has_ceir = false;
    bool consumer_has_ceir = false;
};

enum class program_planning_authorization_status_v1 : std::uint8_t {
    authorized = 0,
    semantic_body_unavailable,
    field_identity_missing,
    authorization_missing,
    source_not_authorized
};

[[nodiscard]] program_planning_authorization_status_v1
authorize_cross_tu_program_planning_v1(
    const cross_tu_planning_request_v1& request,
    const program_planning_authorization_v1& authorization) noexcept;

}  // namespace cellerator::compiler::lto::v1
