__global__ void stage_bucket_finalize_kernel_(
    const float *bucket_sum,
    const float *bucket_sumsq,
    const int *bucket_rows,
    std::int64_t bucket_count,
    float ranking_margin,
    float min_within_day_std,
    int use_neighbor_day_pairs_only,
    std::int64_t num_day_buckets,
    float *bucket_mean,
    float *row_anchor_scale,
    float *row_rank_scale,
    float *spread_row_scale,
    float *ranking_out,
    float *anchor_out,
    float *spread_out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int active_bucket_count = 0;
    for (std::int64_t bucket = 0; bucket < bucket_count; ++bucket) {
        row_anchor_scale[bucket] = 0.0f;
        row_rank_scale[bucket] = 0.0f;
        spread_row_scale[bucket] = 0.0f;
        bucket_mean[bucket] = 0.0f;
        if (bucket_rows[bucket] > 0) ++active_bucket_count;
    }
    if (active_bucket_count == 0) {
        *ranking_out = 0.0f;
        *anchor_out = 0.0f;
        *spread_out = 0.0f;
        return;
    }

    float anchor_sum = 0.0f;
    float spread_sum = 0.0f;
    const std::int64_t total_buckets = num_day_buckets > 0 ? num_day_buckets : bucket_count;

    for (std::int64_t bucket = 0; bucket < bucket_count; ++bucket) {
        const int rows = bucket_rows[bucket];
        if (rows <= 0) continue;
        const float inv_rows = 1.0f / static_cast<float>(rows);
        const float mean = bucket_sum[bucket] * inv_rows;
        const float variance = fmaxf(bucket_sumsq[bucket] * inv_rows - mean * mean, 0.0f);
        const float stddev = sqrtf(variance + kStdEps);
        bucket_mean[bucket] = mean;

        const float anchor_value = total_buckets > 1
            ? static_cast<float>(bucket) / static_cast<float>(total_buckets - 1)
            : 0.5f;
        anchor_sum += (mean - anchor_value) * (mean - anchor_value);
        row_anchor_scale[bucket] = (2.0f * (mean - anchor_value))
            / (static_cast<float>(active_bucket_count) * static_cast<float>(rows));

        if (rows > 1 && min_within_day_std > stddev) {
            spread_sum += min_within_day_std - stddev;
            spread_row_scale[bucket] = -1.0f
                / (static_cast<float>(active_bucket_count) * static_cast<float>(rows) * stddev);
        }
    }

    float ranking_sum = 0.0f;
    int ranking_pairs = 0;
    std::int64_t prev_bucket = -1;
    for (std::int64_t bucket = 0; bucket < bucket_count; ++bucket) {
        if (bucket_rows[bucket] <= 0) continue;
        if (use_neighbor_day_pairs_only) {
            if (prev_bucket >= 0) {
                const float delta = bucket_mean[bucket] - bucket_mean[prev_bucket];
                const float penalty = ranking_margin - delta;
                if (penalty > 0.0f) {
                    ranking_sum += penalty;
                    ++ranking_pairs;
                    row_rank_scale[prev_bucket] += -1.0f;
                    row_rank_scale[bucket] += 1.0f;
                }
            }
            prev_bucket = bucket;
            continue;
        }

        for (std::int64_t other = bucket + 1; other < bucket_count; ++other) {
            if (bucket_rows[other] <= 0) continue;
            const float delta = bucket_mean[other] - bucket_mean[bucket];
            const float penalty = ranking_margin - delta;
            if (penalty > 0.0f) {
                ranking_sum += penalty;
                ++ranking_pairs;
                row_rank_scale[bucket] += -1.0f;
                row_rank_scale[other] += 1.0f;
            }
        }
    }

    if (ranking_pairs > 0) {
        const float inv_pairs = 1.0f / static_cast<float>(ranking_pairs);
        for (std::int64_t bucket = 0; bucket < bucket_count; ++bucket) {
            const int rows = bucket_rows[bucket];
            if (rows > 0) row_rank_scale[bucket] *= inv_pairs / static_cast<float>(rows);
        }
        *ranking_out = ranking_sum * inv_pairs;
    } else {
        *ranking_out = 0.0f;
    }

    *anchor_out = anchor_sum / static_cast<float>(active_bucket_count);
    *spread_out = spread_sum / static_cast<float>(active_bucket_count);
}
