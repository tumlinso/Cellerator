#pragma once

#include <Cellerator/compute/math/operation_core/operation_core.hh>
#include <Cellerator/compute/math/physical_csr.hh>
#include <Cellerator/runtime/session.cuh>

#include <cusparse.h>

#include <cstddef>
#include <cstdint>

namespace cellerator::compute::math::core {

inline constexpr std::uint32_t cusparse_csr_candidate_schema_version = 1u;
inline constexpr stable_id cusparse_csr_spmv_candidate_id{
    0x6375737061727365ull, 0x5f6373725f6d7631ull};
inline constexpr stable_id cusparse_csr_spmm_candidate_id{
    0x6375737061727365ull, 0x5f6373725f6d6d31ull};

enum class cusparse_csr_operation : std::uint8_t {
    spmv = 1u,
    spmm = 2u
};

// Complete candidate-local preparation costs. Projection construction,
// including the explicit f16-to-f32 value-plane conversion required by the
// V100 cuSPARSE type matrix, and transfer remain separate planner costs because
// this candidate consumes an already activated f32 CSR projection.
struct cusparse_csr_preparation_costs {
    std::uint64_t descriptor_state_bytes = 0u;
    std::uint64_t preprocessing_workspace_bytes = 0u;
    std::uint64_t transient_workspace_bytes = 0u;
    std::uint32_t descriptor_create_calls = 0u;
    std::uint32_t preprocess_calls = 0u;
};

// Descriptors and preprocessing storage are created once during preparation.
// The cuSPARSE handle and preprocessing allocation are owned by the execution
// session; clear_cusparse_csr_prepared_state destroys descriptors before the
// session is cleared. Run only rebinds changing value/dense/output pointers.
struct cusparse_csr_prepared_state {
    std::uint32_t schema_version = cusparse_csr_candidate_schema_version;
    cusparse_csr_operation operation = cusparse_csr_operation::spmv;
    std::uint8_t reserved[3]{};
    std::int32_t device_ordinal = -1;
    std::uint32_t dense_columns = 0u;
    execution_csr_view projection{};
    execution::axis_identity feature_axis{};
    execution::axis_identity row_axis{};
    execution::axis_identity column_axis{};
    cudaStream_t prepared_stream = nullptr;
    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t sparse = nullptr;
    cusparseDnVecDescr_t input_vector = nullptr;
    cusparseDnVecDescr_t output_vector = nullptr;
    cusparseDnMatDescr_t input_matrix = nullptr;
    cusparseDnMatDescr_t output_matrix = nullptr;
    void *preprocessing_workspace = nullptr;
    std::size_t preprocessing_workspace_bytes = 0u;
    execution::operand_axis_contract input_contract{};
    execution::operand_axis_contract output_contract{};
    execution::output_axis_contract output_orders[2]{};
    execution::output_effect_contract output_effect{};
    cusparse_csr_preparation_costs costs{};
};

operation_candidate cusparse_csr_spmv_candidate() noexcept;
operation_candidate cusparse_csr_spmm_candidate() noexcept;

operation_status register_cusparse_csr_candidates(
    candidate_registry *registry) noexcept;

// `initial_dense` and `initial_output` are preparation-time device addresses
// used to create persistent dense descriptors. They are not frozen launch
// bindings and may be replaced on every run.
operation_status prepare_cusparse_csr_operation(
    const operation_problem &problem,
    const structure_set_key &structures,
    const projection_key &projection,
    const numeric_policy &numeric,
    const prepare_policy &policy,
    const execution_csr_view &device_csr,
    runtime::execution_session *session,
    std::uint32_t stream_index,
    std::uint32_t dense_columns,
    void *initial_dense,
    void *initial_output,
    execution::axis_identity feature_axis,
    execution::axis_identity row_axis,
    execution::axis_identity column_axis,
    cusparse_csr_prepared_state *state,
    prepared_operation *prepared) noexcept;

void clear_cusparse_csr_prepared_state(
    cusparse_csr_prepared_state *state) noexcept;

cusparse_csr_preparation_costs cusparse_csr_costs(
    const cusparse_csr_prepared_state &state) noexcept;

} // namespace cellerator::compute::math::core
