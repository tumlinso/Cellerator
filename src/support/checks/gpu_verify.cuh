#pragma once

struct gpu_verify_accum {
    unsigned long long samples;
    unsigned long long exact_cast;
    unsigned long long rounded_cast;
    unsigned long long overflow_cast;
    unsigned long long pipeline_match;
    unsigned long long pipeline_mismatch;
    unsigned long long missing_nonzero;
    unsigned long long finite_samples;
    unsigned long long rel_le_001pct;
    unsigned long long rel_le_01pct;
    unsigned long long rel_le_1pct;
    unsigned long long rel_le_5pct;
    unsigned long long rel_gt_5pct;
    double mean_abs_err_sum;
    double mean_rel_err_sum;
    float max_abs_err;
    float max_rel_err;
};
