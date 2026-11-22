#pragma once
#include <stddef.h>
#include <stdint.h>

int workflow() {
    namespace pp = ccell::preprocess;
    namespace v = ccell::velocity;
    namespace mem = cellerator::memory;
    double spliced[4]{1, 2, 3, 4}, unspliced[4]{2, 4, 6, 8};
    mem::const_matrix_view<double> counts{spliced, 4, 1, 1, {mem::domain::host}};
    double mean[1], variance[1];
    uint64_t stats_scratch[1];
    size_t observations = 0;
    auto status = pp::feature_stats(counts, nullptr, mean, variance, 1,
        stats_scratch, 1, &observations);
    if (!status || observations != 4 || mean[0] != 2.5 ||
        __builtin_fabs(variance[0] - 5.0/3) > 1e-12) return 1;

    v::FitOptions options;
    options.trimming = v::Trimming::none;
    double fit_scratch[4];
    auto fit = v::reference::fit_gene(spliced, unspliced, 4, 1, options, fit_scratch);
    return __builtin_fabs(fit.gamma - 2) > 1e-12 ? 2 : 0;
}
