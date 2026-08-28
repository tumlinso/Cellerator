#pragma once

#include <Cellerator/compat/cp_math_v1/operation.hh>

#include <type_traits>

namespace cellerator::compute::math {

inline constexpr u32 execution_plan_schema_version = 1u;

enum class preprocessing_kind : u32 {
    none = 0u,
    backend_preprocess = 1u
};

enum class epilogue_strategy_kind : u32 {
    none = 0u,
    backend_fused = 1u,
    generic_unfused = 2u
};

// Frozen CP-Math v1 evidence only. This pointer-free record remains for
// compatibility/referee tests; supported preparation uses
// operation_core::prepared_operation and execution::launch_bindings.
struct execution_plan {
    u32 schema_version = execution_plan_schema_version;
    u32 physical_view_schema_version = 0u;
    operation_signature operation{};
    u64 backend_identity = 0u;
    u64 algorithm_identity = 0u;
    u64 kernel_variant_identity = 0u;
    u64 workspace_bytes = 0u;
    preprocessing_kind preprocessing = preprocessing_kind::none;
    epilogue_strategy_kind epilogue_strategy = epilogue_strategy_kind::none;
    u64 device_fingerprint = 0u;
    u64 toolchain_fingerprint = 0u;
    u64 tuning_identity = 0u;
};

static_assert(std::is_trivially_copyable<execution_plan>::value,
    "ExecutionPlan must remain pointer-free serializable metadata");
static_assert(std::is_standard_layout<execution_plan>::value,
    "ExecutionPlan must retain a stable standard layout");
static_assert(std::has_unique_object_representations<execution_plan>::value,
    "ExecutionPlan byte serialization must not contain ambiguous padding");

} // namespace cellerator::compute::math
