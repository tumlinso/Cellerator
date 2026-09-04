#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>
namespace cellerator::compiler::lto::v1 {
inline constexpr std::uint32_t companion_artifact_version_v1=1;
enum class object_format_v1:std::uint8_t{elf=1,mach_o,coff,archive,sidecar};
struct artifact_identity_v1{std::uint64_t high=0,low=0;};
struct field_export_v1{artifact_identity_v1 field{};std::string symbol;};
struct provenance_map_v1{artifact_identity_v1 generated{},source{};};
struct ceir_companion_artifact_v1{std::uint32_t version=companion_artifact_version_v1;object_format_v1 format=object_format_v1::elf;artifact_identity_v1 semantic_summary{},planning_summary{},profile_reference{},toolchain{};std::array<std::uint8_t,32>content_hash{};std::vector<field_export_v1>fields;std::vector<provenance_map_v1>provenance;std::string placement;};
enum class companion_status_v1:std::uint8_t{valid=0,unsupported_version,invalid_identity,missing_hash,invalid_placement,duplicate_export};
[[nodiscard]] companion_status_v1 validate_companion_artifact_v1(const ceir_companion_artifact_v1&)noexcept;
}
