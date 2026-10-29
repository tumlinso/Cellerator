#pragma once
#include <Cellerator/compute/operation/operation_core_v2/schema.hh>
#include <cstdint>
namespace cellerator::compiler::profile::v1 {
enum class profile_state_transfer_v1:std::uint8_t{preserve=0,derive,invalidate,unknown};
struct semantic_transfer_v1{profile_state_transfer_v1 values=profile_state_transfer_v1::unknown,support=profile_state_transfer_v1::unknown,structure=profile_state_transfer_v1::unknown,order=profile_state_transfer_v1::unknown,generation=profile_state_transfer_v1::unknown;};
enum native_profile_publication_v1:std::uint32_t{native_publishes_values_v1=1u<<0,native_publishes_structure_v1=1u<<1,native_publishes_order_v1=1u<<2,native_publishes_support_v1=1u<<3};
semantic_transfer_v1 semantic_transfer_for_operation_v1(cellerator::compute::operation::v2::operation_kind) noexcept;
semantic_transfer_v1 semantic_transfer_for_native_effect_v1(std::uint32_t,bool opaque) noexcept;
}  // namespace cellerator::compiler::profile::v1
