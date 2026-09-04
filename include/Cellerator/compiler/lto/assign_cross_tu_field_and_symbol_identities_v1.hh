#pragma once
#include <Cellerator/compiler/lto/freeze_the_ceir_companion_object_artifact_contract_v1.hh>
#include <cstdint>
#include <string>
#include <vector>
namespace cellerator::compiler::lto::v1 {
enum class linkage_v1:std::uint8_t{external=1,weak,hidden,anonymous_namespace};
enum class exported_entity_kind_v1:std::uint8_t{domain=1,relation,field,profile,pass,native_symbol,template_instantiation};
struct cross_tu_symbol_v1{exported_entity_kind_v1 kind=exported_entity_kind_v1::field;std::string name,module;artifact_identity_v1 semantic_fingerprint{};linkage_v1 linkage=linkage_v1::external;};
struct resolved_cross_tu_symbol_v1{cross_tu_symbol_v1 symbol{};artifact_identity_v1 identity{};};
enum class cross_tu_identity_status_v1:std::uint8_t{valid=0,invalid_symbol,odr_conflict,duplicate_strong};
[[nodiscard]] cross_tu_identity_status_v1 assign_cross_tu_identities_v1(const std::vector<cross_tu_symbol_v1>&,std::vector<resolved_cross_tu_symbol_v1>*)noexcept;
}
