__global__ void verify_csr_samples_kernel(matrix::Index samples,
                                          const matrix::Index * __restrict__ row_ptr,
                                          const matrix::Index * __restrict__ col_idx,
                                          const matrix::Real * __restrict__ val,
                                          const matrix::Index * __restrict__ sample_rows,
                                          const matrix::Index * __restrict__ sample_cols,
                                          const int * __restrict__ sample_original,
                                          gpu_verify_accum * __restrict__ stats) {
    const matrix::Index tid = (matrix::Index) (blockIdx.x * blockDim.x + threadIdx.x);
    const matrix::Index stride = (matrix::Index) (gridDim.x * blockDim.x);
    matrix::Index i = tid;

    while (i < samples) {
        const matrix::Index row = sample_rows[i];
        const matrix::Index col = sample_cols[i];
        const int original = sample_original[i];
        const float expected_cast = matrix::real_to_float(matrix::real_from_float((float) original));
        const float original_float = (float) original;
        const matrix::Index begin = row_ptr[row];
        const matrix::Index end = row_ptr[row + 1];
        float stored = 0.0f;
        int found = 0;
        matrix::Index j = begin;

        while (j < end) {
            if (col_idx[j] == col) {
                stored = matrix::real_to_float(val[j]);
                found = 1;
                break;
            }
            ++j;
        }

        atomicAdd(&stats->samples, 1ull);
        if (!found) atomicAdd(&stats->missing_nonzero, 1ull);
        if (stored == expected_cast) atomicAdd(&stats->pipeline_match, 1ull);
        else atomicAdd(&stats->pipeline_mismatch, 1ull);

        if (!isfinite(expected_cast)) {
            atomicAdd(&stats->overflow_cast, 1ull);
            i += stride;
            continue;
        }

        if (expected_cast == original_float) atomicAdd(&stats->exact_cast, 1ull);
        else atomicAdd(&stats->rounded_cast, 1ull);

        {
            const float abs_err = fabsf(expected_cast - original_float);
            const float denom = fabsf(original_float) > 1.0f ? fabsf(original_float) : 1.0f;
            const float rel_err = abs_err / denom;

            atomicAdd(&stats->finite_samples, 1ull);
            atomic_add_double_bits(&stats->mean_abs_err_sum, (double) abs_err);
            atomic_add_double_bits(&stats->mean_rel_err_sum, (double) rel_err);
            atomic_max_float(&stats->max_abs_err, abs_err);
            atomic_max_float(&stats->max_rel_err, rel_err);
            if (rel_err <= 0.00001f) atomicAdd(&stats->rel_le_001pct, 1ull);
            else if (rel_err <= 0.0001f) atomicAdd(&stats->rel_le_01pct, 1ull);
            else if (rel_err <= 0.01f) atomicAdd(&stats->rel_le_1pct, 1ull);
            else if (rel_err <= 0.05f) atomicAdd(&stats->rel_le_5pct, 1ull);
            else atomicAdd(&stats->rel_gt_5pct, 1ull);
        }

        i += stride;
    }
}
