#pragma once
#include <cstdint>
#include <type_traits>
namespace Cellerator::compiler::migration {
struct program_ir_identity_v1{std::uint64_t high=0,low=0;};
enum class program_ir_entity_v1:std::uint8_t{field=1,operation,atom_flow,profile_family};
struct program_ir_record_v1{program_ir_identity_v1 identity{},domain{},order{};program_ir_entity_v1 entity=program_ir_entity_v1::field;std::uint8_t reserved[7]{};std::uint64_t structure_generation=0;};
[[nodiscard]] constexpr bool valid(program_ir_record_v1 r)noexcept{return r.identity.high&&r.identity.low&&r.domain.high&&r.domain.low&&r.order.high&&r.order.low&&r.structure_generation&&r.entity>=program_ir_entity_v1::field&&r.entity<=program_ir_entity_v1::profile_family;}
static_assert(std::is_trivially_copyable_v<program_ir_record_v1>);
} // namespace Cellerator::compiler::migration
