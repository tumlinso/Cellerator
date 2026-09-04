#pragma once
#include <Cellerator/compiler/ir/realization/implement_order_transforms_and_persistent_physical_order_v1.hh>
#include <cstdint>
#include <string>
#include <vector>
namespace cellerator::compiler::ir::realization::v1 {
enum class component_readiness_v1:std::uint8_t{preparing=1,ready};
struct generation_component_v1{stable_identity_v1 identity{};component_readiness_v1 state=component_readiness_v1::preparing;std::uint64_t generation=0;};
struct generation_publication_v1{std::uint64_t current_generation=0,pending_generation=0;order_identity_v1 retained_order{};std::vector<generation_component_v1>components;bool canonicalization_requested=false,published=false;};
enum class generation_publication_status_v1:std::uint8_t{ready=0,invalid_generation,invalid_order,invalid_component,component_pending,partial_publication,already_published};
[[nodiscard]] generation_publication_status_v1 validate_generation_publication_v1(const generation_publication_v1&,std::string*error=nullptr)noexcept;
[[nodiscard]] generation_publication_status_v1 publish_generation_v1(generation_publication_v1*,std::string*error=nullptr)noexcept;
} // namespace cellerator::compiler::ir::realization::v1
