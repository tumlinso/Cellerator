#pragma once
#include <Cellerator/compiler/lto/freeze_the_ceir_companion_object_artifact_contract_v1.hh>
#include <cstdint>
#include <string>
#include <vector>
namespace cellerator::compiler::lto::v1 {
enum class semantic_import_depth_v1:std::uint8_t{summary=1,full_body};
struct exported_semantic_field_v1{artifact_identity_v1 identity{},source{},provenance{};std::string name,summary,body;std::vector<std::string>required_extensions;};
struct semantic_import_request_v1{artifact_identity_v1 field{};semantic_import_depth_v1 depth=semantic_import_depth_v1::summary;std::vector<std::string>supported_extensions;};
struct imported_semantic_field_v1{exported_semantic_field_v1 field{};bool body_loaded=false;};
enum class semantic_import_status_v1:std::uint8_t{valid=0,field_not_found,extension_unsupported,body_unavailable};
[[nodiscard]] semantic_import_status_v1 import_cross_tu_semantic_field_v1(const semantic_import_request_v1&,const std::vector<exported_semantic_field_v1>&,imported_semantic_field_v1*)noexcept;
}
