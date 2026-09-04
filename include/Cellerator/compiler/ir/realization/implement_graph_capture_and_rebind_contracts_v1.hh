#pragma once
#include <Cellerator/compiler/ir/realization/implement_memory_workspace_and_residency_requirements_v1.hh>
#include <cstdint>
#include <string>
#include <vector>
namespace cellerator::compiler::ir::realization::v1 {
enum rebind_field_v1:std::uint32_t{rebind_input_v1=1u<<0,rebind_output_v1=1u<<1,rebind_values_v1=1u<<2,rebind_stream_v1=1u<<3,rebind_generation_v1=1u<<4};
enum class update_policy_owner_v1:std::uint8_t{caller=1,session};
enum class replay_variant_v1:std::uint8_t{exact=1,parameter_update,recapture};
struct graph_stable_binding_v1{stable_identity_v1 symbolic_binding{};stable_identity_v1 address_identity{};bool fixed=false;};
struct graph_capture_contract_v1{stable_identity_v1 identity{};bool capture_compatible=false;std::uint32_t rebindable_fields=0;std::vector<graph_stable_binding_v1>graph_stable_bindings;update_policy_owner_v1 update_owner=update_policy_owner_v1::caller;replay_variant_v1 replay=replay_variant_v1::exact;};
struct graph_rebind_request_v1{std::uint32_t changed_fields=0;std::vector<stable_identity_v1>address_identities;};
enum class graph_capture_status_v1:std::uint8_t{valid=0,not_capture_compatible,invalid_identity,duplicate_binding,fixed_binding_changed,field_not_rebindable,address_count_mismatch,recapture_required};
[[nodiscard]] graph_capture_status_v1 validate_graph_capture_contract_v1(const graph_capture_contract_v1&,std::string*error=nullptr)noexcept;
[[nodiscard]] graph_capture_status_v1 validate_graph_rebind_v1(const graph_capture_contract_v1&,const graph_rebind_request_v1&,std::string*error=nullptr)noexcept;
} // namespace cellerator::compiler::ir::realization::v1
