#pragma once
#include <Cellerator/compiler/profile/represent_domain_axis_relation_and_support_evidence_v1.hh>
#include <cstdint>
#include <type_traits>
namespace cellerator::compiler::profile::v1 {
enum class profile_evidence_location_v1:std::uint8_t{none=0,embedded,external};
struct semantic_ir_profile_attachment_v1{std::uint32_t schema_version=1,reserved=0;profile_identity_v1 semantic_ir_node{},environment{},state{},evidence{};profile_evidence_location_v1 location=profile_evidence_location_v1::none;std::uint8_t reserved8[7]{};std::uint64_t embedded_offset=0,embedded_bytes=0,external_artifact_low=0,external_artifact_high=0;};
enum class semantic_ir_profile_attachment_status_v1:std::uint8_t{ok=0,invalid_identity,invalid_embedded,invalid_external,unsupported_schema};
semantic_ir_profile_attachment_status_v1 validate_semantic_ir_profile_attachment_v1(const semantic_ir_profile_attachment_v1&) noexcept;
static_assert(std::is_trivially_copyable_v<semantic_ir_profile_attachment_v1>);
}  // namespace cellerator::compiler::profile::v1
