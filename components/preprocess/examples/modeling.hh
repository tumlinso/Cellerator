#pragma once

// Included by the native and AnnData consumers after importing cellerator.preprocess.
namespace ccell_example {

struct ExpressionModel {
    double weight[2];

    double operator()(cellerator::memory::const_matrix_view<double> x, size_t cell) const {
        const double* row = x.data + cell*x.row_stride;
        return weight[0]*row[0] + weight[1]*row[x.column_stride];
    }
};

template<class Stride>
bool normalized_readout(cellerator::memory::const_matrix_view<float, Stride> counts) {
    using namespace cellerator::memory;
    using cellerator::preprocess::NormalizationBuffers;
    using cellerator::preprocess::NormalizationOptions;
    if (counts.rows != 2 || counts.columns != 2) return false;

    double storage[4], factors[2], scratch[2];
    uint8_t included[2];
    const matrix_view<double> normalized{storage, 2, 2, 2, {domain::host}};
    const NormalizationOptions options{.target_is_set = true, .target_sum = 8};
    NormalizationBuffers buffers{factors, 2, included, 2, scratch, 2};
    if (!cellerator::preprocess::normalize_into(counts, normalized, options, buffers)) return false;

    const ExpressionModel model{{1, 2}};
    const auto expression = as_const(normalized);
    return storage[0] == 2 && storage[1] == 6 && storage[2] == 6 && storage[3] == 2 &&
        model(expression, 0) == 14 && model(expression, 1) == 10;
}

}
