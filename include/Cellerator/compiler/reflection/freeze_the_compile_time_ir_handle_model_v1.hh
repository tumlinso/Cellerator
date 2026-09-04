#pragma once
#include <cstdint>
namespace cellerator::compiler::reflection::v1 {
enum class handle_kind_v1:std::uint8_t{source_declaration=1,field,operation,relation,profile_state,planning_alternative,selected_realization,provenance};
enum class availability_phase_v1:std::uint8_t{source=1,semantic,profiled,planned,realized};
enum class handle_lifetime_v1:std::uint8_t{expression=1,translation_unit,compilation,artifact};
struct ir_handle_v1{std::uint64_t identity_high=0,identity_low=0,arena_epoch=0,object_generation=0;handle_kind_v1 kind=handle_kind_v1::source_declaration;availability_phase_v1 available_at=availability_phase_v1::source;handle_lifetime_v1 lifetime=handle_lifetime_v1::compilation;};
struct handle_context_v1{std::uint64_t arena_epoch=0;availability_phase_v1 phase=availability_phase_v1::source;};
enum class handle_status_v1:std::uint8_t{valid=0,invalid_identity,unavailable,expired,stale};
[[nodiscard]] handle_status_v1 validate_handle_v1(const ir_handle_v1&,const handle_context_v1&,std::uint64_t current_generation)noexcept;
[[nodiscard]] ir_handle_v1 preserve_handle_for_safe_transform_v1(const ir_handle_v1&)noexcept;
[[nodiscard]] ir_handle_v1 invalidate_handle_for_edit_v1(const ir_handle_v1&)noexcept;
} // namespace cellerator::compiler::reflection::v1
