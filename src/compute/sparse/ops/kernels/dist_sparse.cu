#include "../ops.hh"

namespace cellerator::compute::sparse::ops::dist {

namespace {

runtime::execution_context slot_context_(const runtime::fleet_context &fleet, unsigned int slot) {
    runtime::execution_context ctx;
    ctx.device = runtime::fleet_device_id(fleet, slot);
    ctx.stream = runtime::fleet_stream(fleet, slot);
    ctx.owns_stream = false;
    return ctx;
}

void require_slots_(const runtime::fleet_context &fleet, const unsigned int *slots, unsigned int slot_count) {
    if (slots == nullptr && slot_count != 0) throw std::invalid_argument("distributed launch requires slot storage");
    for (unsigned int i = 0; i < slot_count; ++i) {
        if (!runtime::fleet_slot_available(fleet, slots[i])) throw std::out_of_range("distributed launch slot is unavailable");
    }
}

} // namespace

void reduce_sum_to_leader_f32(
    runtime::fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const float *const *partials,
    std::size_t count,
    float *leader_out) {
    runtime::reduce_sum_to_leader_f32(fleet, slots, slot_count, partials, count, slot_count != 0u ? slots[0] : 0u, leader_out);
}

void launch_csr_row_scale_fwd_f16_f32(
    runtime::fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const std::uint32_t *const *major_ptr,
    const __half *const *values,
    const float *const *row_scales,
    const std::uint32_t *rows,
    float *const *out_values) {
    if (fleet == nullptr) throw std::invalid_argument("launch_csr_row_scale_fwd_f16_f32 requires a fleet");
    require_slots_(*fleet, slots, slot_count);
    for (unsigned int i = 0; i < slot_count; ++i) {
        const runtime::execution_context ctx = slot_context_(*fleet, slots[i]);
        base::csr_row_scale_fwd_f16_f32(ctx, major_ptr[i], values[i], row_scales[i], rows[i], out_values[i]);
    }
}

void launch_csr_row_scale_bwd_values_f16_f32(
    runtime::fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const std::uint32_t *const *major_ptr,
    const float *const *grad_out,
    const float *const *row_scales,
    const std::uint32_t *rows,
    float *const *grad_values) {
    if (fleet == nullptr) throw std::invalid_argument("launch_csr_row_scale_bwd_values_f16_f32 requires a fleet");
    require_slots_(*fleet, slots, slot_count);
    for (unsigned int i = 0; i < slot_count; ++i) {
        const runtime::execution_context ctx = slot_context_(*fleet, slots[i]);
        base::csr_row_scale_bwd_values_f16_f32(ctx, major_ptr[i], grad_out[i], row_scales[i], rows[i], grad_values[i]);
    }
}

void launch_csr_row_scale_bwd_scales_f16_f32(
    runtime::fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const std::uint32_t *const *major_ptr,
    const __half *const *values,
    const float *const *grad_out,
    const std::uint32_t *rows,
    float *const *grad_scales) {
    if (fleet == nullptr) throw std::invalid_argument("launch_csr_row_scale_bwd_scales_f16_f32 requires a fleet");
    require_slots_(*fleet, slots, slot_count);
    for (unsigned int i = 0; i < slot_count; ++i) {
        const runtime::execution_context ctx = slot_context_(*fleet, slots[i]);
        base::csr_row_scale_bwd_scales_f16_f32(ctx, major_ptr[i], values[i], grad_out[i], rows[i], grad_scales[i]);
    }
}

void launch_sparse_value_sum_f16_f32(
    runtime::fleet_context *fleet,
    runtime::scratch_arena *scratch_per_slot,
    const unsigned int *slots,
    unsigned int slot_count,
    const __half *const *values,
    const std::uint32_t *nnz,
    float *const *out_scalar) {
    if (fleet == nullptr) throw std::invalid_argument("launch_sparse_value_sum_f16_f32 requires a fleet");
    if (scratch_per_slot == nullptr && slot_count != 0) throw std::invalid_argument("launch_sparse_value_sum_f16_f32 requires per-slot scratch");
    require_slots_(*fleet, slots, slot_count);
    for (unsigned int i = 0; i < slot_count; ++i) {
        const runtime::execution_context ctx = slot_context_(*fleet, slots[i]);
        base::sparse_value_sum_f16_f32(ctx, scratch_per_slot + i, values[i], nnz[i], out_scalar[i]);
    }
}

void launch_csr_spmv_fwd_f16_f32(
    runtime::fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const std::uint32_t *const *major_ptr,
    const std::uint32_t *const *minor_idx,
    const __half *const *values,
    const std::uint32_t *rows,
    const float *const *vector,
    float *const *out) {
    if (fleet == nullptr) throw std::invalid_argument("launch_csr_spmv_fwd_f16_f32 requires a fleet");
    require_slots_(*fleet, slots, slot_count);
    for (unsigned int i = 0; i < slot_count; ++i) {
        const runtime::execution_context ctx = slot_context_(*fleet, slots[i]);
        base::csr_spmv_fwd_f16_f32(ctx, major_ptr[i], minor_idx[i], values[i], rows[i], vector[i], out[i]);
    }
}

void launch_csr_spmv_bwd_values_f16_f32(
    runtime::fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const std::uint32_t *const *major_ptr,
    const std::uint32_t *const *minor_idx,
    const float *const *grad_out,
    const float *const *vector,
    const std::uint32_t *rows,
    float *const *grad_values) {
    if (fleet == nullptr) throw std::invalid_argument("launch_csr_spmv_bwd_values_f16_f32 requires a fleet");
    require_slots_(*fleet, slots, slot_count);
    for (unsigned int i = 0; i < slot_count; ++i) {
        const runtime::execution_context ctx = slot_context_(*fleet, slots[i]);
        base::csr_spmv_bwd_values_f16_f32(ctx, major_ptr[i], minor_idx[i], grad_out[i], vector[i], rows[i], grad_values[i]);
    }
}

void launch_csr_spmv_bwd_vector_f16_f32(
    runtime::fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const std::uint32_t *const *major_ptr,
    const std::uint32_t *const *minor_idx,
    const __half *const *values,
    const float *const *grad_out,
    const std::uint32_t *rows,
    const std::uint32_t *cols,
    float *const *grad_vector) {
    if (fleet == nullptr) throw std::invalid_argument("launch_csr_spmv_bwd_vector_f16_f32 requires a fleet");
    require_slots_(*fleet, slots, slot_count);
    for (unsigned int i = 0; i < slot_count; ++i) {
        const runtime::execution_context ctx = slot_context_(*fleet, slots[i]);
        base::csr_spmv_bwd_vector_f16_f32(ctx, major_ptr[i], minor_idx[i], values[i], grad_out[i], rows[i], cols[i], grad_vector[i]);
    }
}

void launch_csr_spmm_bwd_values_f16_f32(
    runtime::fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const std::uint32_t *const *major_ptr,
    const std::uint32_t *const *minor_idx,
    const float *const *grad_out,
    const float *const *rhs,
    const std::uint32_t *rows,
    const std::int64_t *rhs_ld,
    const std::int64_t *out_cols,
    float *const *grad_values) {
    if (fleet == nullptr) throw std::invalid_argument("launch_csr_spmm_bwd_values_f16_f32 requires a fleet");
    require_slots_(*fleet, slots, slot_count);
    for (unsigned int i = 0; i < slot_count; ++i) {
        const runtime::execution_context ctx = slot_context_(*fleet, slots[i]);
        base::csr_spmm_bwd_values_f16_f32(ctx, major_ptr[i], minor_idx[i], grad_out[i], rhs[i], rows[i], rhs_ld[i], out_cols[i], grad_values[i]);
    }
}

void launch_csr_spmm_bwd_rhs_f16_f32(
    runtime::fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const std::uint32_t *const *major_ptr,
    const std::uint32_t *const *minor_idx,
    const __half *const *values,
    const float *const *grad_out,
    const std::uint32_t *rows,
    const std::uint32_t *cols,
    const std::int64_t *grad_out_ld,
    const std::int64_t *out_cols,
    float *const *grad_rhs,
    const std::int64_t *grad_rhs_ld) {
    if (fleet == nullptr) throw std::invalid_argument("launch_csr_spmm_bwd_rhs_f16_f32 requires a fleet");
    require_slots_(*fleet, slots, slot_count);
    for (unsigned int i = 0; i < slot_count; ++i) {
        const runtime::execution_context ctx = slot_context_(*fleet, slots[i]);
        base::csr_spmm_bwd_rhs_f16_f32(ctx, major_ptr[i], minor_idx[i], values[i], grad_out[i], rows[i], cols[i], grad_out_ld[i], out_cols[i], grad_rhs[i], grad_rhs_ld[i]);
    }
}

} // namespace cellerator::compute::sparse::ops::dist
