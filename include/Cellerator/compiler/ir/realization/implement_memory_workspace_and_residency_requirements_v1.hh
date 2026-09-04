#pragma once
#include <Cellerator/compiler/ir/realization/implement_physical_plane_representation_v1.hh>
#include <cstdint>
#include <string>
#include <vector>
namespace cellerator::compiler::ir::realization::v1 {
enum class allocation_class_v1:std::uint8_t{persistent=1,transient,graph_stable};
enum class allocation_owner_v1:std::uint8_t{caller=1,session};
struct memory_requirement_v1{stable_identity_v1 identity{};allocation_class_v1 allocation=allocation_class_v1::transient;allocation_owner_v1 owner=allocation_owner_v1::caller;address_space_class_v1 address_space=address_space_class_v1::host;plane_lifetime_v1 lifetime=plane_lifetime_v1::invocation;std::uint64_t bytes=0,capacity_bytes=0;std::uint32_t alignment=1;};
struct session_memory_accounting_v1{std::uint64_t host_capacity=0,device_capacity=0,persistent_used=0,graph_stable_used=0,transient_available=0;};
enum class memory_requirement_status_v1:std::uint8_t{valid=0,invalid_identity,invalid_capacity,invalid_alignment,invalid_lifetime,insufficient_host,insufficient_device,insufficient_transient};
[[nodiscard]] memory_requirement_status_v1 validate_memory_requirements_v1(const std::vector<memory_requirement_v1>&,std::string*error=nullptr) noexcept;
[[nodiscard]] memory_requirement_status_v1 compare_memory_requirements_v1(const std::vector<memory_requirement_v1>&,const session_memory_accounting_v1&,std::string*error=nullptr) noexcept;
} // namespace cellerator::compiler::ir::realization::v1
