#pragma once

#include <Cellerator/types.cuh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::compute::math {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

inline constexpr u32 operation_contract_schema_version = 1u;
inline constexpr u32 feature_order_identity_schema_version = 1u;
inline constexpr u32 sparse_structure_identity_schema_version = 1u;

enum class operation_kind : u32 {
    spmm = 1u
};

enum class transpose_kind : u32 {
    none = 0u,
    transpose = 1u
};

enum class dense_layout_kind : u32 {
    row_major = 1u,
    column_major = 2u
};

enum class determinism_requirement : u32 {
    allow_nondeterministic = 0u,
    deterministic = 1u
};

enum class workspace_policy_kind : u32 {
    reusable_pool = 1u,
    caller_limit = 2u,
    no_additional_workspace = 3u
};

enum class expected_reuse_kind : u32 {
    single_run = 1u,
    bounded = 2u,
    persistent = 3u
};

enum class epilogue_kind : u32 {
    none = 0u,
    bias = 1u,
    relu = 2u,
    gelu_exact_erf = 3u,
    gelu_tanh_approximate = 4u,
    bias_relu = 5u,
    bias_gelu_exact_erf = 6u,
    bias_gelu_tanh_approximate = 7u
};

// Exact GELU is 0.5*x*(1+erf(x/sqrt(2))). The tanh approximation is
// 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))). They are distinct semantics.

enum class feature_order_kind : u32 {
    canonical = 1u,
    packed = 2u
};

enum class request_validation_code : u32 {
    ok = 0u,
    unsupported_version = 1u,
    invalid_shape = 2u,
    invalid_type = 3u,
    invalid_scalar = 4u,
    invalid_workspace_policy = 5u,
    invalid_reuse = 6u,
    invalid_feature_order = 7u,
    feature_order_mismatch = 8u,
    missing_bias = 9u,
    unexpected_bias = 10u,
    invalid_layout = 11u,
    invalid_identity = 12u,
    missing_binding = 13u,
    invalid_determinism = 14u,
    invalid_epilogue = 15u
};

// Feature-axis identity is semantic. Packed order additionally binds the exact
// CP-BP feature-block geometry; pointer addresses never participate.
struct feature_order_identity {
    u32 schema_version = feature_order_identity_schema_version;
    feature_order_kind kind = feature_order_kind::canonical;
    u32 feature_count = 0u;
    u32 feature_axis_identity_version = 0u;
    u64 feature_axis_identity = 0u;
    u64 packing_geometry_identity = 0u;
};

// Stable sparse structure identity is part of planner/cache identity. It is
// derived from durable structure or a PackingPlan, never from a live pointer.
struct sparse_structure_identity {
    u32 schema_version = sparse_structure_identity_schema_version;
    u32 identity_version = 0u;
    u64 value = 0u;
};

struct scalar_value {
    u32 type_code = 0u;
    u32 reserved = 0u;
    u64 bits = 0u;
};

struct workspace_policy {
    workspace_policy_kind kind = workspace_policy_kind::reusable_pool;
    u32 reserved = 0u;
    // Nonzero only for caller_limit; other policies have no numeric cap.
    u64 byte_limit = 0u;
};

struct expected_reuse {
    expected_reuse_kind kind = expected_reuse_kind::single_run;
    u32 reserved = 0u;
    // single_run=1, bounded=finite nonzero, persistent=0 (unbounded).
    u64 expected_run_count = 1u;
};

struct epilogue_descriptor {
    epilogue_kind kind = epilogue_kind::none;
    u32 bias_type_code = 0u;
    u64 bias_element_count = 0u;
};

// Backend-neutral mathematical request. Physical view, algorithm, kernel, and
// handle selection are deliberately absent. Bindings are kept separate so the
// signature and plan-cache identity never depend on live addresses. M/K/N are
// the logical dimensions after applying the declared operand transposes.
struct spmm_request {
    u32 schema_version = operation_contract_schema_version;
    operation_kind operation = operation_kind::spmm;
    u64 m = 0u;
    u64 k = 0u;
    u64 n = 0u;
    u64 sparse_nnz = 0u;
    sparse_structure_identity sparse_structure{};
    transpose_kind transpose_sparse = transpose_kind::none;
    transpose_kind transpose_dense = transpose_kind::none;
    dense_layout_kind dense_rhs_layout = dense_layout_kind::row_major;
    dense_layout_kind output_layout = dense_layout_kind::row_major;
    u64 dense_rhs_leading_dimension = 0u;
    u64 output_leading_dimension = 0u;
    u32 sparse_storage_type_code = 0u;
    u32 dense_storage_type_code = 0u;
    u32 output_storage_type_code = 0u;
    u32 compute_type_code = 0u;
    u32 accumulation_type_code = 0u;
    scalar_value alpha{};
    scalar_value beta{};
    determinism_requirement determinism =
        determinism_requirement::allow_nondeterministic;
    workspace_policy workspace{};
    expected_reuse reuse{};
    epilogue_descriptor epilogue{};
    feature_order_identity sparse_feature_order{};
    feature_order_identity dense_feature_order{};
};

struct spmm_bindings {
    const void *sparse_matrix = nullptr;
    const void *dense_rhs = nullptr;
    void *output = nullptr;
    const void *bias = nullptr;
    u64 sparse_matrix_identity = 0u;
    u64 dense_rhs_identity = 0u;
    u64 output_identity = 0u;
};

struct math_request {
    spmm_request operation{};
    spmm_bindings bindings{};
};

struct operation_signature {
    u32 schema_version = operation_contract_schema_version;
    operation_kind operation = operation_kind::spmm;
    u64 low = 0u;
    u64 high = 0u;
};

struct request_validation_result {
    request_validation_code code = request_validation_code::ok;
    const char *message = "ok";

    constexpr explicit operator bool() const noexcept {
        return code == request_validation_code::ok;
    }
};

enum class trivial_operation_kind : u32 {
    none = 0u,
    no_output = 1u,
    epilogue_only = 2u
};

scalar_value make_scalar(float value) noexcept;
scalar_value make_scalar(double value) noexcept;
bool scalar_is_zero(const scalar_value &value) noexcept;
bool same_feature_order(
    const feature_order_identity &lhs,
    const feature_order_identity &rhs) noexcept;
request_validation_result validate_spmm_request(const spmm_request &request) noexcept;
request_validation_result validate_math_request(const math_request &request) noexcept;
operation_signature make_operation_signature(const spmm_request &request) noexcept;
trivial_operation_kind classify_trivial_operation(const spmm_request &request) noexcept;

static_assert(std::is_trivially_copyable<feature_order_identity>::value,
    "feature order identity must remain serializable");
static_assert(std::is_trivially_copyable<sparse_structure_identity>::value,
    "sparse structure identity must remain serializable");
static_assert(std::is_trivially_copyable<spmm_request>::value,
    "SpMM request metadata must remain serializable");
static_assert(std::is_trivially_copyable<operation_signature>::value,
    "operation signature must remain serializable");

} // namespace cellerator::compute::math
