#include "../autograd.hh"

namespace cellerator::compute::autograd::dist {

namespace {

constexpr int kAddThreads = 256;
constexpr int kInvalidPairScore = -1000000000;

struct pair_reduction_plan {
    unsigned int peer0_index = 0u;
    unsigned int leader1_index = 0u;
    unsigned int peer1_index = 0u;
    unsigned int peer0 = 0u;
    unsigned int leader1 = 0u;
    unsigned int peer1 = 0u;
    bool valid = false;
};

__global__ void dense_add_inplace_kernel_(float *dst, const float *src, std::size_t count) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    dst[idx] += src[idx];
}

execution_context slot_context_(const fleet_context &fleet, unsigned int slot) {
    execution_context ctx;
    ctx.device = fleet_device_id(fleet, slot);
    ctx.stream = fleet_stream(fleet, slot);
    ctx.owns_stream = false;
    return ctx;
}

void require_slots_(const fleet_context &fleet, const unsigned int *slots, unsigned int slot_count) {
    if (slots == nullptr && slot_count != 0) throw std::invalid_argument("distributed launch requires slot storage");
    for (unsigned int i = 0; i < slot_count; ++i) {
        if (!fleet_slot_available(fleet, slots[i])) throw std::out_of_range("distributed launch slot is unavailable");
    }
}

void copy_or_alias_to_leader_(
    fleet_context *fleet,
    unsigned int leader_slot,
    const float *src,
    int src_device,
    std::size_t count,
    float *leader_out) {
    const int leader_device = fleet_device_id(*fleet, leader_slot);
    cuda_require(cudaSetDevice(leader_device), "cudaSetDevice(copy_or_alias_to_leader)");
    if (src == leader_out && src_device == leader_device) return;
    const std::size_t bytes = count * sizeof(float);
    if (src_device == leader_device) {
        cuda_require(cudaMemcpyAsync(leader_out, src, bytes, cudaMemcpyDeviceToDevice, fleet_stream(*fleet, leader_slot)), "cudaMemcpyAsync(copy_or_alias_to_leader)");
        return;
    }
    cuda_require(cudaMemcpyPeerAsync(leader_out, leader_device, src, src_device, bytes, fleet_stream(*fleet, leader_slot)), "cudaMemcpyPeerAsync(copy_or_alias_to_leader)");
}

int pair_link_score_(const fleet_context &fleet, unsigned int lhs_slot, unsigned int rhs_slot) {
    const int lhs_rank = fleet_peer_performance_rank(fleet, lhs_slot, rhs_slot);
    const int rhs_rank = fleet_peer_performance_rank(fleet, rhs_slot, lhs_slot);
    if (lhs_rank < 0 || rhs_rank < 0) return kInvalidPairScore;
    return -(lhs_rank + rhs_rank);
}

bool matches_native_slot_order_(const fleet_context &fleet, const unsigned int *slots, unsigned int slot_count) {
    if (slot_count != 4u || !fleet_has_native_v100_topology(fleet)) return false;
    unsigned int native_slots[4] = {};
    if (default_native_fleet_slots(fleet, native_slots, 4u) != 4u) return false;
    for (unsigned int i = 0; i < 4u; ++i) {
        if (slots[i] != native_slots[i]) return false;
    }
    return true;
}

pair_reduction_plan build_generic_pair_plan_(const fleet_context &fleet, const unsigned int *slots, unsigned int slot_count) {
    pair_reduction_plan best_plan;
    if (slot_count != 4u || slots == nullptr) return best_plan;

    int best_score = kInvalidPairScore;
    for (unsigned int peer_index = 1u; peer_index < 4u; ++peer_index) {
        const unsigned int leader0 = slots[0];
        const unsigned int peer0 = slots[peer_index];
        unsigned int remaining[2] = {};
        unsigned int cursor = 0u;
        for (unsigned int i = 1u; i < 4u; ++i) {
            if (i == peer_index) continue;
            remaining[cursor++] = slots[i];
        }
        const int score0 = pair_link_score_(fleet, leader0, peer0);
        const int score1 = pair_link_score_(fleet, remaining[0], remaining[1]);
        if (score0 == kInvalidPairScore || score1 == kInvalidPairScore) continue;
        const int total_score = score0 + score1;
        if (!best_plan.valid || total_score > best_score) {
            best_score = total_score;
            best_plan.peer0_index = peer_index;
            best_plan.leader1_index = 0u;
            best_plan.peer1_index = 0u;
            best_plan.peer0 = peer0;
            best_plan.leader1 = remaining[0];
            best_plan.peer1 = remaining[1];
            for (unsigned int i = 1u; i < 4u; ++i) {
                if (slots[i] == remaining[0]) best_plan.leader1_index = i;
                if (slots[i] == remaining[1]) best_plan.peer1_index = i;
            }
            best_plan.valid = true;
        }
    }
    return best_plan;
}

} // namespace

void reduce_sum_to_leader_f32(
    fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const float *const *partials,
    std::size_t count,
    float *leader_out) {
    if (fleet == nullptr) throw std::invalid_argument("reduce_sum_to_leader_f32 requires a fleet");
    if (partials == nullptr && slot_count != 0) throw std::invalid_argument("reduce_sum_to_leader_f32 requires partial storage");
    if (leader_out == nullptr && count != 0) throw std::invalid_argument("reduce_sum_to_leader_f32 requires a leader output");
    require_slots_(*fleet, slots, slot_count);
    if (slot_count == 0 || count == 0) return;

    for (unsigned int i = 0; i < slot_count; ++i) {
        const unsigned int slot = slots[i];
        cuda_require(cudaSetDevice(fleet_device_id(*fleet, slot)), "cudaSetDevice(reduce_sum_to_leader sync)");
        if (fleet_stream(*fleet, slot) != nullptr) {
            cuda_require(cudaStreamSynchronize(fleet_stream(*fleet, slot)), "cudaStreamSynchronize(reduce_sum_to_leader)");
        }
    }

    const int blocks = static_cast<int>((count + static_cast<std::size_t>(kAddThreads) - 1u) / static_cast<std::size_t>(kAddThreads));
    const unsigned int leader0 = slots[0];
    const int leader0_device = fleet_device_id(*fleet, leader0);
    copy_or_alias_to_leader_(fleet, leader0, partials[0], fleet_device_id(*fleet, leader0), count, leader_out);

    pair_reduction_plan pair_plan;
    if (slot_count == 4u) {
        if (!build::cuda_mode_is_generic && matches_native_slot_order_(*fleet, slots, slot_count)) {
            pair_plan.peer0_index = 1u;
            pair_plan.leader1_index = 2u;
            pair_plan.peer1_index = 3u;
            pair_plan.peer0 = slots[1];
            pair_plan.leader1 = slots[2];
            pair_plan.peer1 = slots[3];
            pair_plan.valid = true;
        } else {
            pair_plan = build_generic_pair_plan_(*fleet, slots, slot_count);
        }
    }

    if (pair_plan.valid) {
        const unsigned int peer0 = pair_plan.peer0;
        const unsigned int leader1 = pair_plan.leader1;
        const unsigned int peer1 = pair_plan.peer1;

        float *leader0_scratch = static_cast<float *>(request_fleet_scratch(fleet, leader0, count * sizeof(float)));
        cuda_require(
            cudaMemcpyPeerAsync(
                leader0_scratch,
                leader0_device,
                partials[pair_plan.peer0_index],
                fleet_device_id(*fleet, peer0),
                count * sizeof(float),
                fleet_stream(*fleet, leader0)),
            "cudaMemcpyPeerAsync(reduce_sum_to_leader pair0)");
        dense_add_inplace_kernel_<<<blocks, kAddThreads, 0, fleet_stream(*fleet, leader0)>>>(leader_out, leader0_scratch, count);
        cuda_require(cudaGetLastError(), "dense_add_inplace_kernel(reduce_sum_to_leader pair0)");

        const int leader1_device = fleet_device_id(*fleet, leader1);
        float *leader1_scratch = static_cast<float *>(request_fleet_scratch(fleet, leader1, count * sizeof(float) * 2u));
        float *pair1_accum = leader1_scratch;
        float *pair1_tmp = leader1_scratch + count;
        copy_or_alias_to_leader_(fleet, leader1, partials[pair_plan.leader1_index], leader1_device, count, pair1_accum);
        cuda_require(
            cudaMemcpyPeerAsync(
                pair1_tmp,
                leader1_device,
                partials[pair_plan.peer1_index],
                fleet_device_id(*fleet, peer1),
                count * sizeof(float),
                fleet_stream(*fleet, leader1)),
            "cudaMemcpyPeerAsync(reduce_sum_to_leader pair1)");
        dense_add_inplace_kernel_<<<blocks, kAddThreads, 0, fleet_stream(*fleet, leader1)>>>(pair1_accum, pair1_tmp, count);
        cuda_require(cudaGetLastError(), "dense_add_inplace_kernel(reduce_sum_to_leader pair1)");
        cuda_require(cudaStreamSynchronize(fleet_stream(*fleet, leader1)), "cudaStreamSynchronize(reduce_sum_to_leader pair1)");

        cuda_require(cudaSetDevice(leader0_device), "cudaSetDevice(reduce_sum_to_leader leaders)");
        cuda_require(
            cudaMemcpyPeerAsync(leader0_scratch, leader0_device, pair1_accum, leader1_device, count * sizeof(float), fleet_stream(*fleet, leader0)),
            "cudaMemcpyPeerAsync(reduce_sum_to_leader leaders)");
        dense_add_inplace_kernel_<<<blocks, kAddThreads, 0, fleet_stream(*fleet, leader0)>>>(leader_out, leader0_scratch, count);
        cuda_require(cudaGetLastError(), "dense_add_inplace_kernel(reduce_sum_to_leader leaders)");
        return;
    }

    for (unsigned int i = 1; i < slot_count; ++i) {
        const unsigned int slot = slots[i];
        float *scratch = static_cast<float *>(request_fleet_scratch(fleet, leader0, count * sizeof(float)));
        cuda_require(cudaSetDevice(leader0_device), "cudaSetDevice(reduce_sum_to_leader direct)");
        cuda_require(
            cudaMemcpyPeerAsync(scratch, leader0_device, partials[i], fleet_device_id(*fleet, slot), count * sizeof(float), fleet_stream(*fleet, leader0)),
            "cudaMemcpyPeerAsync(reduce_sum_to_leader direct)");
        dense_add_inplace_kernel_<<<blocks, kAddThreads, 0, fleet_stream(*fleet, leader0)>>>(leader_out, scratch, count);
        cuda_require(cudaGetLastError(), "dense_add_inplace_kernel(reduce_sum_to_leader direct)");
    }
}

void launch_csr_row_scale_fwd_f16_f32(
    fleet_context *fleet,
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
        const execution_context ctx = slot_context_(*fleet, slots[i]);
        base::csr_row_scale_fwd_f16_f32(ctx, major_ptr[i], values[i], row_scales[i], rows[i], out_values[i]);
    }
}

void launch_csr_row_scale_bwd_values_f16_f32(
    fleet_context *fleet,
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
        const execution_context ctx = slot_context_(*fleet, slots[i]);
        base::csr_row_scale_bwd_values_f16_f32(ctx, major_ptr[i], grad_out[i], row_scales[i], rows[i], grad_values[i]);
    }
}

void launch_csr_row_scale_bwd_scales_f16_f32(
    fleet_context *fleet,
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
        const execution_context ctx = slot_context_(*fleet, slots[i]);
        base::csr_row_scale_bwd_scales_f16_f32(ctx, major_ptr[i], values[i], grad_out[i], rows[i], grad_scales[i]);
    }
}

void launch_sparse_value_sum_f16_f32(
    fleet_context *fleet,
    scratch_arena *scratch_per_slot,
    const unsigned int *slots,
    unsigned int slot_count,
    const __half *const *values,
    const std::uint32_t *nnz,
    float *const *out_scalar) {
    if (fleet == nullptr) throw std::invalid_argument("launch_sparse_value_sum_f16_f32 requires a fleet");
    if (scratch_per_slot == nullptr && slot_count != 0) throw std::invalid_argument("launch_sparse_value_sum_f16_f32 requires per-slot scratch");
    require_slots_(*fleet, slots, slot_count);
    for (unsigned int i = 0; i < slot_count; ++i) {
        const execution_context ctx = slot_context_(*fleet, slots[i]);
        base::sparse_value_sum_f16_f32(ctx, scratch_per_slot + i, values[i], nnz[i], out_scalar[i]);
    }
}

void launch_csr_spmv_fwd_f16_f32(
    fleet_context *fleet,
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
        const execution_context ctx = slot_context_(*fleet, slots[i]);
        base::csr_spmv_fwd_f16_f32(ctx, major_ptr[i], minor_idx[i], values[i], rows[i], vector[i], out[i]);
    }
}

void launch_csr_spmv_bwd_values_f16_f32(
    fleet_context *fleet,
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
        const execution_context ctx = slot_context_(*fleet, slots[i]);
        base::csr_spmv_bwd_values_f16_f32(ctx, major_ptr[i], minor_idx[i], grad_out[i], vector[i], rows[i], grad_values[i]);
    }
}

void launch_csr_spmv_bwd_vector_f16_f32(
    fleet_context *fleet,
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
        const execution_context ctx = slot_context_(*fleet, slots[i]);
        base::csr_spmv_bwd_vector_f16_f32(ctx, major_ptr[i], minor_idx[i], values[i], grad_out[i], rows[i], cols[i], grad_vector[i]);
    }
}

void launch_csr_spmm_fwd_f16_f32(
    fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const std::uint32_t *const *major_ptr,
    const std::uint32_t *const *minor_idx,
    const __half *const *values,
    const std::uint32_t *rows,
    const std::uint32_t *cols,
    const float *const *rhs,
    const std::int64_t *rhs_ld,
    const std::int64_t *out_cols,
    float *const *out,
    const std::int64_t *out_ld) {
    if (fleet == nullptr) throw std::invalid_argument("launch_csr_spmm_fwd_f16_f32 requires a fleet");
    require_slots_(*fleet, slots, slot_count);
    for (unsigned int i = 0; i < slot_count; ++i) {
        const execution_context ctx = slot_context_(*fleet, slots[i]);
        base::csr_spmm_fwd_f16_f32(ctx, major_ptr[i], minor_idx[i], values[i], rows[i], cols[i], rhs[i], rhs_ld[i], out_cols[i], out[i], out_ld[i]);
    }
}

void launch_blocked_ell_spmm_fwd_f16_f32_lib(
    fleet_context *fleet,
    cusparse_cache *cache_per_slot,
    const unsigned int *slots,
    unsigned int slot_count,
    const void *const *matrix_token,
    const std::uint32_t *const *block_col_idx,
    const __half *const *values,
    const std::uint32_t *rows,
    const std::uint32_t *cols,
    const std::uint32_t *block_size,
    const std::uint32_t *ell_cols,
    const float *const *rhs,
    const std::int64_t *rhs_ld,
    const std::int64_t *out_cols,
    float *const *out,
    const std::int64_t *out_ld) {
    if (fleet == nullptr) throw std::invalid_argument("launch_blocked_ell_spmm_fwd_f16_f32_lib requires a fleet");
    if (cache_per_slot == nullptr && slot_count != 0u) throw std::invalid_argument("launch_blocked_ell_spmm_fwd_f16_f32_lib requires cache storage");
    require_slots_(*fleet, slots, slot_count);
    for (unsigned int i = 0; i < slot_count; ++i) {
        const execution_context ctx = slot_context_(*fleet, slots[i]);
        base::blocked_ell_spmm_fwd_f16_f32_lib(
            ctx,
            cache_per_slot + i,
            matrix_token != nullptr ? matrix_token[i] : values[i],
            block_col_idx[i],
            values[i],
            rows[i],
            cols[i],
            block_size[i],
            ell_cols[i],
            rhs[i],
            rhs_ld[i],
            out_cols[i],
            out[i],
            out_ld[i]);
    }
}

void launch_blocked_ell_spmm_fwd_f16_f16_f32_lib(
    fleet_context *fleet,
    cusparse_cache *cache_per_slot,
    const unsigned int *slots,
    unsigned int slot_count,
    const void *const *matrix_token,
    const std::uint32_t *const *block_col_idx,
    const __half *const *values,
    const std::uint32_t *rows,
    const std::uint32_t *cols,
    const std::uint32_t *block_size,
    const std::uint32_t *ell_cols,
    const __half *const *rhs,
    const std::int64_t *rhs_ld,
    const std::int64_t *out_cols,
    float *const *out,
    const std::int64_t *out_ld) {
    if (fleet == nullptr) throw std::invalid_argument("launch_blocked_ell_spmm_fwd_f16_f16_f32_lib requires a fleet");
    if (cache_per_slot == nullptr && slot_count != 0u) throw std::invalid_argument("launch_blocked_ell_spmm_fwd_f16_f16_f32_lib requires cache storage");
    require_slots_(*fleet, slots, slot_count);
    for (unsigned int i = 0; i < slot_count; ++i) {
        const execution_context ctx = slot_context_(*fleet, slots[i]);
        base::blocked_ell_spmm_fwd_f16_f16_f32_lib(
            ctx,
            cache_per_slot + i,
            matrix_token != nullptr ? matrix_token[i] : values[i],
            block_col_idx[i],
            values[i],
            rows[i],
            cols[i],
            block_size[i],
            ell_cols[i],
            rhs[i],
            rhs_ld[i],
            out_cols[i],
            out[i],
            out_ld[i]);
    }
}

void launch_csr_spmm_bwd_values_f16_f32(
    fleet_context *fleet,
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
        const execution_context ctx = slot_context_(*fleet, slots[i]);
        base::csr_spmm_bwd_values_f16_f32(ctx, major_ptr[i], minor_idx[i], grad_out[i], rhs[i], rows[i], rhs_ld[i], out_cols[i], grad_values[i]);
    }
}

void launch_csr_spmm_bwd_rhs_f16_f32(
    fleet_context *fleet,
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
        const execution_context ctx = slot_context_(*fleet, slots[i]);
        base::csr_spmm_bwd_rhs_f16_f32(ctx, major_ptr[i], minor_idx[i], values[i], grad_out[i], rows[i], cols[i], grad_out_ld[i], out_cols[i], grad_rhs[i], grad_rhs_ld[i]);
    }
}

} // namespace cellerator::compute::autograd::dist
