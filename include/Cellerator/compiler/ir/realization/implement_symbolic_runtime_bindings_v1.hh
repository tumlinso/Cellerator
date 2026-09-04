#pragma once
#include <Cellerator/compiler/ir/realization/freeze_realization_ir_module_and_target_scopes_v1.hh>
#include <cstdint>
#include <type_traits>
#include <vector>
namespace cellerator::compiler::ir::realization::v1 {
enum class binding_slot_kind_v1:std::uint8_t{input=1,output,values,workspace,native_symbol,stream,handle,event,lease};
struct symbolic_binding_slot_v1{stable_identity_v1 identity{};binding_slot_kind_v1 kind=binding_slot_kind_v1::input;std::uint8_t reserved[7]{};std::uint64_t minimum_bytes=0,alignment=1;};
struct symbolic_binding_table_v1{stable_identity_v1 identity{};std::vector<symbolic_binding_slot_v1>slots;};
struct live_runtime_binding_v1{stable_identity_v1 slot{};void*address=nullptr;void*stream=nullptr;std::uint64_t runtime_token=0;};
enum class symbolic_binding_status_v1:std::uint8_t{valid=0,invalid_identity,duplicate_slot,invalid_alignment,missing_live_binding,unexpected_live_binding,null_live_resource};
[[nodiscard]] symbolic_binding_status_v1 validate_symbolic_binding_table_v1(const symbolic_binding_table_v1&)noexcept;
[[nodiscard]] symbolic_binding_status_v1 bind_symbolic_runtime_v1(const symbolic_binding_table_v1&,const std::vector<live_runtime_binding_v1>&)noexcept;
[[nodiscard]] std::vector<std::uint8_t> serialize_symbolic_bindings_v1(const symbolic_binding_table_v1&);
static_assert(std::is_trivially_copyable_v<symbolic_binding_slot_v1>);
} // namespace cellerator::compiler::ir::realization::v1
