#pragma once

#include <Cellerator/compute/operation/relation_algebra.hh>
#include <Cellerator/compute/operation/relation_algebra_v2/relation_algebra.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::operation::v2 {

enum class compatibility_execution_authority : std::uint8_t {
    operation_core_v2 = 2
};

struct v1_adapter_storage {
    typed_relation *relations = nullptr;
    relation_binding_contract *bindings = nullptr;
    relation_value_binding_contract *value_bindings = nullptr;
    std::uint64_t capacity = 0;
};

struct v1_adapter_request {
    stable_id persistent_problem_identity{};
    execution::value_generation value_generation{};
    v1_adapter_storage storage{};
};

struct v1_adapter_result {
    relation_algebra_problem problem{};
    compatibility_execution_authority authority =
        compatibility_execution_authority::operation_core_v2;
    bool source_only_compatibility = true;
    std::uint8_t reserved[6]{};
};

schema_status adapt_relation_algebra_v1(
    const relation_algebra_problem_v1 &source,
    const v1_adapter_request &request,
    v1_adapter_result *result) noexcept;

static_assert(std::is_trivially_copyable_v<v1_adapter_storage>);
static_assert(std::is_trivially_copyable_v<v1_adapter_request>);
static_assert(std::is_trivially_copyable_v<v1_adapter_result>);

}  // namespace cellerator::compute::operation::v2
