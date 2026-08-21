#pragma once

#include <Cellerator/execution/execution_contract.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::compute::math::core {

inline constexpr std::uint32_t operation_core_schema_version = 1u;
inline constexpr std::uint32_t operation_candidate_capacity = 64u;

struct stable_id {
    std::uint64_t low = 0u;
    std::uint64_t high = 0u;
};

enum class operation_kind : std::uint16_t {
    sparse_dense_multiply = 1u,
    weighted_relation_reduce = 2u,
    sequence_predicate_accumulate = 3u
};

enum class projection_kind : std::uint16_t {
    native_row_masked = 1u,
    native_feature_major = 2u,
    cta_macrotile = 3u,
    dense_fragment = 4u,
    csr = 5u,
    sell = 6u,
    bsr = 7u,
    blocked_ell = 8u,
    vendor_specific = 9u,
    transpose_or_backward = 10u,
    architecture_specific = 11u
};

enum class backend_kind : std::uint8_t {
    native_direct = 1u,
    vendor_library = 2u,
    composed = 3u
};

enum class rounding_policy : std::uint8_t {
    nearest_even = 1u,
    toward_zero = 2u,
    stochastic = 3u
};

enum class saturation_policy : std::uint8_t {
    none = 1u,
    saturate = 2u
};

enum class quantization_granularity : std::uint8_t {
    none = 1u,
    value_plane = 2u,
    module = 3u,
    block = 4u
};

enum candidate_capability_flag : std::uint32_t {
    candidate_deterministic = 1u << 0u,
    candidate_graph_capture = 1u << 1u,
    candidate_persistent_preprocessing = 1u << 2u,
    candidate_composed_epilogue = 1u << 3u
};

struct operation_problem {
    std::uint32_t schema_version = operation_core_schema_version;
    operation_kind kind = operation_kind::sparse_dense_multiply;
    std::uint16_t reserved = 0u;
    stable_id operation{};
    std::uint32_t input_count = 0u;
    std::uint32_t output_count = 0u;
    std::uint64_t logical_work_items = 0u;
};

// Semantic structure identity and its hot runtime binding are both explicit.
// Neither is derived from a pointer.
struct structure_key {
    execution::structure_id persistent{};
    execution::structure_handle runtime{};
    execution::structure_epoch epoch{};
};

struct projection_key {
    execution::projection_id persistent{};
    execution::projection_handle runtime{};
    projection_kind kind = projection_kind::native_row_masked;
    std::uint16_t schema_version = 0u;
    std::uint32_t variant = 0u;
};

struct numeric_policy {
    execution::numeric_type sparse_storage = execution::numeric_type::invalid;
    execution::numeric_type dense_storage = execution::numeric_type::invalid;
    execution::numeric_type output_storage = execution::numeric_type::invalid;
    execution::numeric_type multiply = execution::numeric_type::invalid;
    execution::numeric_type accumulation = execution::numeric_type::invalid;
    execution::numeric_type scalar = execution::numeric_type::invalid;
    execution::numeric_type bias = execution::numeric_type::invalid;
    rounding_policy rounding = rounding_policy::nearest_even;
    saturation_policy saturation = saturation_policy::none;
    quantization_granularity quantization = quantization_granularity::none;
    std::uint8_t reserved[5]{};
};

struct prepare_policy {
    bool deterministic = false;
    bool graph_capture_required = false;
    bool allow_persistent_preprocessing = true;
    bool allow_composed_epilogue = true;
    std::uint32_t expected_reuse = 1u;
    std::uint64_t persistent_memory_limit = 0u;
    std::uint64_t transient_memory_limit = 0u;
};

struct persistent_kernel_state {
    const void *data = nullptr;
    std::uint64_t bytes = 0u;
};

struct prepared_operation;

enum class operation_status_code : std::uint8_t {
    ok = 0u,
    invalid_argument = 1u,
    unsupported_problem = 2u,
    unsupported_numeric_policy = 3u,
    unsupported_projection = 4u,
    capability_rejected = 5u,
    registry_full = 6u,
    duplicate_candidate = 7u,
    stale_structure = 8u,
    invalid_launch_bindings = 9u,
    preparation_failed = 10u,
    execution_failed = 11u
};

struct operation_status {
    operation_status_code code = operation_status_code::ok;
    execution::binding_validation_code binding =
        execution::binding_validation_code::ok;
    const char *message = "ok";

    constexpr explicit operator bool() const noexcept {
        return code == operation_status_code::ok;
    }
};

using run_function = operation_status (*)(
    const prepared_operation &,
    const execution::launch_bindings &) noexcept;

// Reusable state freezes semantics, immutable structure, projection, algorithm,
// persistent state, transient requirements, and output-order contracts. All
// changing pointers, values, scalars, streams, and transient workspace remain
// in execution::launch_bindings.
struct prepared_operation {
    std::uint32_t schema_version = operation_core_schema_version;
    operation_problem problem{};
    structure_key structure{};
    projection_key projection{};
    numeric_policy numeric{};
    stable_id kernel{};
    backend_kind backend = backend_kind::native_direct;
    std::uint8_t reserved[7]{};
    std::uint32_t capability_flags = 0u;
    std::uint32_t reserved_flags = 0u;
    persistent_kernel_state persistent{};
    execution::prepared_binding_contract binding_contract{};
    run_function run = nullptr;
};

struct operation_candidate;

using numeric_support_function = bool (*)(const numeric_policy &) noexcept;
using prepare_function = operation_status (*)(
    const operation_candidate &,
    const operation_problem &,
    const structure_key &,
    const projection_key &,
    const numeric_policy &,
    const prepare_policy &,
    prepared_operation *) noexcept;

struct operation_candidate {
    stable_id identity{};
    const char *name = nullptr;
    operation_kind operation = operation_kind::sparse_dense_multiply;
    projection_kind projection = projection_kind::native_row_masked;
    backend_kind backend = backend_kind::native_direct;
    std::uint8_t reserved[3]{};
    std::uint32_t capability_flags = 0u;
    std::uint64_t persistent_bytes = 0u;
    std::uint64_t transient_bytes = 0u;
    numeric_support_function supports_numeric = nullptr;
    prepare_function prepare = nullptr;
};

struct candidate_registry {
    operation_candidate candidates[operation_candidate_capacity]{};
    std::uint32_t size = 0u;
};

operation_status validate_operation_problem(
    const operation_problem &problem,
    const structure_key &structure) noexcept;
operation_status validate_numeric_policy(const numeric_policy &numeric) noexcept;
operation_status validate_prepared_operation(
    const prepared_operation &prepared) noexcept;
operation_status run_prepared_operation(
    const prepared_operation &prepared,
    const execution::launch_bindings &launch) noexcept;

operation_status register_candidate(
    candidate_registry *registry,
    const operation_candidate &candidate) noexcept;
const operation_candidate *find_candidate(
    const candidate_registry &registry,
    stable_id identity) noexcept;
operation_status prepare_candidate(
    const operation_candidate &candidate,
    const operation_problem &problem,
    const structure_key &structure,
    const projection_key &projection,
    const numeric_policy &numeric,
    const prepare_policy &policy,
    prepared_operation *prepared) noexcept;

constexpr bool same_stable_id(stable_id lhs, stable_id rhs) noexcept {
    return lhs.low == rhs.low && lhs.high == rhs.high;
}

static_assert(std::is_trivially_copyable<operation_problem>::value,
    "operation problem must remain cache-key safe");
static_assert(std::is_trivially_copyable<structure_key>::value,
    "structure key must remain cache-key safe");
static_assert(std::is_trivially_copyable<projection_key>::value,
    "projection key must remain cache-key safe");
static_assert(std::is_trivially_copyable<numeric_policy>::value,
    "numeric policy must remain cache-key safe");
static_assert(std::is_trivially_copyable<prepared_operation>::value,
    "prepared operation is a compact dispatch record");

} // namespace cellerator::compute::math::core
