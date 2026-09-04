#pragma once
#include <Cellerator/compiler/lto/freeze_the_ceir_companion_object_artifact_contract_v1.hh>
#include <cstdint>
#include <vector>
namespace cellerator::compiler::lto::v1 {
struct thin_lto_identity_v1{artifact_identity_v1 semantic{},profile{},toolchain{},passes{};};
struct thin_lto_object_v1{artifact_identity_v1 field{};thin_lto_identity_v1 identity{};std::vector<artifact_identity_v1>calls;bool full_ceir_cached=false;};
struct incremental_lto_plan_v1{std::vector<artifact_identity_v1>reused_summaries,reused_full_ceir,replan_fields;};
[[nodiscard]] incremental_lto_plan_v1 plan_incremental_thin_lto_v1(const std::vector<thin_lto_object_v1>&current,const std::vector<thin_lto_object_v1>&cache)noexcept;
}
