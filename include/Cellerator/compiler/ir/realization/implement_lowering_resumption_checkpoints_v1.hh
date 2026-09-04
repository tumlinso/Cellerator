#pragma once
#include <Cellerator/compiler/ir/realization/freeze_realization_ir_module_and_target_scopes_v1.hh>
#include <array>
#include <cstdint>
#include <string>
#include <vector>
namespace cellerator::compiler::ir::realization::v1 {
enum class ceir_facet_v1:std::uint8_t{canonical_source=0,atom_evidence,semantic_atom,target_cover,physical_projection,packed_operand,executable_recipe,local_realization,count};
struct lowering_checkpoint_v1{ceir_facet_v1 facet=ceir_facet_v1::canonical_source;stable_identity_v1 artifact{};stable_identity_v1 content_hash{};stable_identity_v1 input_hash{};std::uint64_t structure_epoch=0,value_generation=0;};
struct lowering_resumption_plan_v1{std::array<bool,static_cast<std::size_t>(ceir_facet_v1::count)>reusable{};ceir_facet_v1 resume_at=ceir_facet_v1::canonical_source;};
enum class lowering_checkpoint_status_v1:std::uint8_t{valid=0,invalid_identity,missing_facet,duplicate_facet,content_corrupt,input_changed,epoch_changed,generation_changed};
[[nodiscard]] lowering_checkpoint_status_v1 plan_lowering_resumption_v1(const std::vector<lowering_checkpoint_v1>&stored,const std::vector<lowering_checkpoint_v1>&expected,lowering_resumption_plan_v1*,std::string*error=nullptr)noexcept;
} // namespace cellerator::compiler::ir::realization::v1
