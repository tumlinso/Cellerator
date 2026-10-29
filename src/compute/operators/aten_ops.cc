#include <Cellerator/compute/operators/aten_ops.hh>

#include <ATen/ATen.h>
#include <ATen/ops/argsort.h>
#include <ATen/ops/bincount.h>
#include <ATen/ops/cumsum.h>
#include <ATen/ops/isfinite.h>
#include <ATen/ops/repeat_interleave.h>
#include <ATen/ops/sparse_csr_tensor.h>

#include <algorithm>
#include <cmath>
#include <c10/core/ScalarType.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace cellerator::aten_ops {
namespace {

at::ScalarType computation_type(
    at::ScalarType input_type,
    const std::optional<at::ScalarType> &requested) {
    if (requested) {
        if (*requested != at::kFloat && *requested != at::kDouble)
            throw std::invalid_argument("compute_type must be float32 or float64");
        return *requested;
    }
    return input_type == at::kDouble ? at::kDouble : at::kFloat;
}

void require_real_numeric(const at::Tensor &tensor, const char *name) {
    if (!tensor.defined()) throw std::invalid_argument(std::string(name) + " is undefined");
    if (tensor.is_complex()) throw std::invalid_argument(std::string(name) + " must be real-valued");
    if (!(tensor.is_floating_point()
        || c10::isIntegralType(tensor.scalar_type(), false)))
        throw std::invalid_argument(std::string(name) + " must have a real numeric dtype");
}

bool all_true(const at::Tensor &condition) {
    return condition.all().item<bool>();
}

void require_finite(const at::Tensor &tensor, const char *name) {
    if (tensor.is_floating_point() && !all_true(at::isfinite(tensor)))
        throw std::invalid_argument(std::string(name) + " must contain only finite values");
}

at::Tensor row_indices(const at::Tensor &csr) {
    const auto crow = csr.crow_indices();
    const auto counts = crow.narrow(0, 1, csr.size(0))
        - crow.narrow(0, 0, csr.size(0));
    return at::repeat_interleave(counts, std::optional<int64_t>{csr._nnz()});
}

struct MatrixData {
    int64_t rows;
    int64_t columns;
    bool sparse;
    at::Tensor values;
    at::Tensor column_indices;
    at::Tensor row_indices;
};

MatrixData matrix_data(const at::Tensor &matrix) {
    validate_feature_matrix(matrix);
    if (matrix.is_sparse_csr()) {
        return {matrix.size(0), matrix.size(1), true, matrix.values(),
            matrix.col_indices(), row_indices(matrix)};
    }
    return {matrix.size(0), matrix.size(1), false, matrix,
        at::Tensor(), at::Tensor()};
}

at::Tensor source_columns(const MatrixData &data) {
    if (data.sparse) return data.column_indices;
    return at::arange(data.columns, data.values.options().dtype(at::kLong))
        .unsqueeze(0).expand({data.rows, data.columns}).reshape({-1});
}

at::Tensor source_rows(const MatrixData &data) {
    if (data.sparse) return data.row_indices;
    return at::arange(data.rows, data.values.options().dtype(at::kLong))
        .unsqueeze(1).expand({data.rows, data.columns}).reshape({-1});
}

at::Tensor flat_values(const MatrixData &data) {
    return data.sparse ? data.values : data.values.reshape({-1});
}

void require_aligned_optional(
    const at::Tensor &tensor,
    const at::Device &device,
    at::ScalarType expected_type,
    at::IntArrayRef expected_shape,
    const char *name) {
    if (tensor.device() != device)
        throw std::invalid_argument(std::string(name) + " must be on the input device");
    if (tensor.scalar_type() != expected_type)
        throw std::invalid_argument(std::string(name) + " has an unsupported dtype");
    if (!tensor.sizes().equals(expected_shape))
        throw std::invalid_argument(std::string(name) + " has an incompatible shape");
}

} // namespace

void validate_feature_matrix(const at::Tensor &matrix) {
    require_real_numeric(matrix, "feature matrix");
    if (matrix.dim() != 2)
        throw std::invalid_argument("feature matrix must have rank two");
    if (matrix.is_sparse_csr()) {
        const auto crow = matrix.crow_indices();
        const auto columns = matrix.col_indices();
        const auto values = matrix.values();
        if (crow.device() != matrix.device() || columns.device() != matrix.device()
            || values.device() != matrix.device())
            throw std::invalid_argument("CSR components must share a device");
        if (crow.dim() != 1 || crow.numel() != matrix.size(0) + 1
            || columns.dim() != 1 || values.dim() != 1
            || columns.numel() != values.numel())
            throw std::invalid_argument("CSR components have invalid shapes");
        if (!(crow.scalar_type() == at::kLong || crow.scalar_type() == at::kInt)
            || !(columns.scalar_type() == at::kLong || columns.scalar_type() == at::kInt))
            throw std::invalid_argument("CSR indices must be int32 or int64");
        if (values.scalar_type() != matrix.scalar_type())
            throw std::invalid_argument("CSR values have an unexpected dtype");
        const auto crow64 = crow.to(at::kLong);
        const auto columns64 = columns.to(at::kLong);
        if (crow64.numel() == 0 || crow64[0].item<int64_t>() != 0
            || crow64[-1].item<int64_t>() != values.numel())
            throw std::invalid_argument("CSR row pointers do not delimit the values");
        if (!all_true(crow64.narrow(0, 1, matrix.size(0)).ge(
                crow64.narrow(0, 0, matrix.size(0)))) )
            throw std::invalid_argument("CSR row pointers must be monotonic");
        if (columns64.numel() > 0
            && (!all_true(columns64.ge(0)) || !all_true(columns64.lt(matrix.size(1)))) )
            throw std::invalid_argument("CSR columns are outside the matrix shape");
        if (columns64.numel() > 1) {
            const auto rows = row_indices(matrix);
            const auto adjacent = rows.numel() - 1;
            const auto same_row = rows.narrow(0, 1, adjacent).eq(
                rows.narrow(0, 0, adjacent));
            const auto increasing = columns64.narrow(0, 1, adjacent).gt(
                columns64.narrow(0, 0, adjacent));
            if (!all_true(same_row.logical_not() | increasing))
                throw std::invalid_argument("CSR columns must be strictly increasing within each row");
        }
    } else if (matrix.layout() != at::kStrided) {
        throw std::invalid_argument("feature matrix must use dense strided or CSR layout");
    }
}

at::Tensor normalize_features(
    const at::Tensor &values,
    std::optional<at::Tensor> row_totals,
    std::optional<at::Tensor> feature_factors,
    double target_sum,
    std::optional<at::ScalarType> compute_type) {
    if (!std::isfinite(target_sum))
        throw std::invalid_argument("target_sum must be finite");
    const auto data = matrix_data(values);
    const auto arithmetic_type = computation_type(values.scalar_type(), compute_type);
    auto input = values.is_sparse_csr() ? values.values() : values;
    auto output = input.to(arithmetic_type);

    if (row_totals) {
        require_aligned_optional(*row_totals, values.device(), row_totals->scalar_type(),
            {data.rows}, "row_totals");
        require_real_numeric(*row_totals, "row_totals");
        auto factors = row_totals->to(arithmetic_type);
        require_finite(factors, "row_totals");
        if (!all_true(factors.ne(0)))
            throw std::invalid_argument("row_totals must be nonzero");
        if (data.sparse) {
            auto rows = data.row_indices;
            output = output / factors.index_select(0, rows);
        } else {
            output = output / factors.unsqueeze(1);
        }
    }
    output = output * target_sum;
    if (feature_factors) {
        require_aligned_optional(*feature_factors, values.device(),
            feature_factors->scalar_type(), {data.columns}, "feature_factors");
        require_real_numeric(*feature_factors, "feature_factors");
        auto factors = feature_factors->to(arithmetic_type);
        require_finite(factors, "feature_factors");
        if (!all_true(factors.ne(0)))
            throw std::invalid_argument("feature_factors must be nonzero");
        if (data.sparse) {
            output = output / factors.index_select(0, data.column_indices);
        } else {
            output = output / factors.unsqueeze(0);
        }
    }

    if (values.is_floating_point()) output = output.to(values.scalar_type());
    if (!values.is_sparse_csr()) return output;
    return at::sparse_csr_tensor(values.crow_indices(), values.col_indices(), output,
        values.sizes(), values.options().layout(at::kSparseCsr).dtype(output.scalar_type()));
}

RankedSelection select_ranked_features(
    const at::Tensor &scores,
    const at::Tensor &feature_ids,
    std::optional<at::Tensor> eligibility,
    int64_t max_per_row) {
    const auto data = matrix_data(scores);
    if (max_per_row < -1)
        throw std::invalid_argument("max_per_row must be -1 or nonnegative");
    if (feature_ids.dim() != 1 || feature_ids.numel() != data.columns
        || feature_ids.scalar_type() != at::kLong || feature_ids.device() != scores.device())
        throw std::invalid_argument("feature_ids must be an int64 vector aligned to columns and device");
    auto ordered_feature_ids = std::get<0>(at::sort(feature_ids, true, 0, false));
    if (ordered_feature_ids.numel() > 1
        && !all_true(ordered_feature_ids.narrow(0, 1, ordered_feature_ids.numel() - 1).ne(
            ordered_feature_ids.narrow(0, 0, ordered_feature_ids.numel() - 1))))
        throw std::invalid_argument("feature_ids must be unique");

    auto values = flat_values(data);
    require_finite(values, "scores");
    auto rows = source_rows(data);
    auto columns = source_columns(data);
    auto source_indices = at::arange(values.numel(), values.options().dtype(at::kLong));
    auto keep = at::ones({values.numel()}, values.options().dtype(at::kBool));
    if (eligibility) {
        if (eligibility->device() != scores.device() || eligibility->scalar_type() != at::kBool)
            throw std::invalid_argument("eligibility must be a bool tensor on the score device");
        if (data.sparse) {
            if (eligibility->dim() != 1 || eligibility->numel() != values.numel())
                throw std::invalid_argument("CSR eligibility must align with scores.values()");
            keep = *eligibility;
        } else {
            if (!eligibility->sizes().equals(scores.sizes()))
                throw std::invalid_argument("dense eligibility must match the score shape");
            keep = eligibility->reshape({-1});
        }
    }

    auto positions = at::nonzero(keep).reshape({-1});
    auto selected_rows = rows.index_select(0, positions);
    auto selected_columns = columns.index_select(0, positions);
    auto selected_values = values.index_select(0, positions);
    auto selected_features = feature_ids.index_select(0, selected_columns);
    // Three stable passes establish the specified lexicographic order.
    auto order = at::argsort(selected_features, true, 0, false);
    order = order.index_select(0, at::argsort(
        selected_values.index_select(0, order), true, 0, true));
    order = order.index_select(0, at::argsort(
        selected_rows.index_select(0, order), true, 0, false));
    auto sorted_rows = selected_rows.index_select(0, order);
    auto sorted_indices = source_indices.index_select(0, positions).index_select(0, order);
    auto counts = at::bincount(sorted_rows, std::nullopt, data.rows);
    if (max_per_row >= 0) {
        auto prior = at::cumsum(counts, 0) - counts;
        auto prior_for_entry = at::repeat_interleave(
            prior, counts, std::nullopt, sorted_rows.numel());
        auto rank = at::arange(sorted_rows.numel(), sorted_rows.options()) - prior_for_entry;
        auto retained = rank.lt(max_per_row);
        sorted_indices = sorted_indices.masked_select(retained);
        sorted_rows = sorted_rows.masked_select(retained);
    }
    auto output_counts = at::bincount(sorted_rows, std::nullopt, data.rows);
    auto zero = at::zeros({1}, output_counts.options());
    auto offsets = at::cat({zero, at::cumsum(output_counts, 0)}, 0);
    return {sorted_indices, offsets};
}

FeatureStatistics nonzero_feature_median(
    const std::vector<at::Tensor> &batches,
    int64_t feature_block_size,
    std::optional<at::ScalarType> compute_type) {
    if (batches.empty())
        throw std::invalid_argument("at least one feature batch is required");
    if (feature_block_size <= 0)
        throw std::invalid_argument("feature_block_size must be positive");
    for (const auto &batch : batches) validate_feature_matrix(batch);
    const auto &first = batches.front();
    const auto columns = first.size(1);
    const auto device = first.device();
    auto dtype = computation_type(first.scalar_type(), compute_type);
    for (const auto &batch : batches) {
        if (batch.size(1) != columns || batch.device() != device)
            throw std::invalid_argument("feature batches must share width and device");
        const auto values = batch.is_sparse_csr() ? batch.values() : batch;
        require_finite(values, "feature batch");
        if (!compute_type && batch.scalar_type() == at::kDouble) dtype = at::kDouble;
    }

    auto options = first.options().layout(at::kStrided).dtype(dtype);
    auto medians = at::full({columns}, std::numeric_limits<double>::quiet_NaN(), options);
    auto counts = at::zeros({columns}, first.options().layout(at::kStrided).dtype(at::kLong));
    auto valid = at::zeros({columns}, first.options().layout(at::kStrided).dtype(at::kBool));

    for (int64_t begin = 0; begin < columns; begin += feature_block_size) {
        const auto width = std::min(feature_block_size, columns - begin);
        std::vector<at::Tensor> block_columns;
        std::vector<at::Tensor> block_values;
        for (const auto &batch : batches) {
            if (batch.is_sparse_csr()) {
                const auto indices = batch.col_indices().to(at::kLong);
                auto in_block = indices.ge(begin) & indices.lt(begin + width)
                    & batch.values().ne(0);
                auto positions = at::nonzero(in_block).reshape({-1});
                block_columns.push_back(indices.index_select(0, positions) - begin);
                block_values.push_back(batch.values().index_select(0, positions).to(dtype));
            } else {
                auto flat = batch.narrow(1, begin, width).reshape({-1});
                auto positions = at::nonzero(flat.ne(0)).reshape({-1});
                block_columns.push_back(at::remainder(positions, width));
                block_values.push_back(flat.index_select(0, positions).to(dtype));
            }
        }
        auto cols = at::cat(block_columns, 0);
        auto vals = at::cat(block_values, 0);
        if (cols.numel() == 0) continue;
        // Stable value sort followed by stable feature sort gives each feature
        // group in ascending value order while keeping exact ties harmless.
        auto order = at::argsort(vals, true, 0, false);
        order = order.index_select(0, at::argsort(
            cols.index_select(0, order), true, 0, false));
        auto sorted_cols = cols.index_select(0, order);
        auto sorted_vals = vals.index_select(0, order);
        auto block_counts = at::bincount(sorted_cols, std::nullopt, width);
        auto starts = at::cumsum(block_counts, 0) - block_counts;
        auto lower_index = starts + at::floor((block_counts - 1).to(at::kDouble) / 2).to(at::kLong);
        auto upper_index = starts + at::floor(block_counts.to(at::kDouble) / 2).to(at::kLong);
        auto safe_lower = lower_index.clamp(0, sorted_vals.numel() - 1);
        auto safe_upper = upper_index.clamp(0, sorted_vals.numel() - 1);
        auto lower_values = sorted_vals.index_select(0, safe_lower);
        auto upper_values = sorted_vals.index_select(0, safe_upper);
        auto same_sign = lower_values.lt(0).eq(upper_values.lt(0));
        // Avoid overflow when the two finite values have the same sign. Keep
        // the direct average for opposite signs so cancellation is accurate.
        auto same_sign_midpoint = lower_values + (upper_values - lower_values) * 0.5;
        auto opposite_sign_midpoint = (lower_values + upper_values) * 0.5;
        auto block_median = at::where(same_sign, same_sign_midpoint,
            opposite_sign_midpoint);
        auto block_valid = block_counts.gt(0);
        block_median = at::where(block_valid, block_median,
            at::full({width}, std::numeric_limits<double>::quiet_NaN(), options));
        medians.slice(0, begin, begin + width).copy_(block_median);
        counts.slice(0, begin, begin + width).copy_(block_counts);
        valid.slice(0, begin, begin + width).copy_(block_valid);
    }
    return {medians, counts, valid};
}

} // namespace cellerator::aten_ops
