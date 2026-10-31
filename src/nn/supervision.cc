#include <Cellerator/nn/supervision.hh>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace cellerator::nn::detail {

std::pair<at::Tensor, at::Tensor> reduce_prediction_loss_impl(
    const at::Tensor& losses,
    SupervisionUnit unit,
    const std::optional<at::Tensor>& valid_mask,
    const std::vector<std::int64_t>& row_owner_indices,
    std::int64_t owner_count) {
    if (!losses.defined() || !losses.is_floating_point()) {
        throw std::invalid_argument("losses must be a defined floating-point tensor");
    }
    if (unit != SupervisionUnit::observation && unit != SupervisionUnit::cell) {
        throw std::invalid_argument("unit must be SupervisionUnit::observation or cell");
    }

    at::Tensor mask;
    if (valid_mask) {
        if (!valid_mask->defined() || valid_mask->scalar_type() != at::kBool) {
            throw std::invalid_argument("valid_mask must be a defined Boolean tensor");
        }
        if (valid_mask->sizes() != losses.sizes() || valid_mask->device() != losses.device()) {
            throw std::invalid_argument("valid_mask must match loss shape and device");
        }
        mask = *valid_mask;
    } else {
        mask = at::ones_like(losses, losses.options().dtype(at::kBool));
    }
    const auto safe = at::where(mask, losses, at::zeros_like(losses));

    if (unit == SupervisionUnit::observation) {
        const auto count = mask.sum(at::kLong);
        const auto denominator = count.clamp_min(1).to(losses.scalar_type());
        auto mean = safe.sum() / denominator;
        return {mean + safe.sum() * 0.0, count};
    }

    const std::int64_t rows = losses.dim() == 0 ? 1 : losses.size(0);
    if (owner_count < 0 || row_owner_indices.size() != static_cast<std::size_t>(rows)) {
        throw std::invalid_argument("cell owner mapping must identify every loss row");
    }
    for (auto owner : row_owner_indices) {
        if (owner < 0 || owner >= owner_count) {
            throw std::out_of_range("cell owner mapping is outside the declared owner set");
        }
    }
    std::int64_t terms_per_row = 1;
    if (losses.dim() > 1) {
        for (std::int64_t axis = 1; axis < losses.dim(); ++axis) {
            terms_per_row *= losses.size(axis);
        }
    }

    std::vector<std::int64_t> term_owners;
    term_owners.reserve(static_cast<std::size_t>(rows * terms_per_row));
    for (auto owner : row_owner_indices) {
        for (std::int64_t term = 0; term < terms_per_row; ++term) {
            term_owners.push_back(owner);
        }
    }
    const auto owner_index = at::tensor(term_owners,
        at::TensorOptions().dtype(at::kLong).device(losses.device()));
    auto sums = at::zeros({owner_count}, losses.options());
    auto counts = at::zeros({owner_count}, losses.options().dtype(at::kLong));
    const auto flat_safe = safe.reshape({-1});
    const auto flat_mask = mask.reshape({-1}).to(at::kLong);
    if (!term_owners.empty()) {
        sums = sums.index_add(0, owner_index, flat_safe);
        counts = counts.index_add(0, owner_index, flat_mask);
    }
    const auto valid_owners = counts > 0;
    const auto cell_means = sums / counts.clamp_min(1).to(losses.scalar_type());
    const auto valid_count = valid_owners.sum(at::kLong);
    const auto mean = at::where(valid_owners, cell_means, at::zeros_like(cell_means)).sum() /
        valid_count.clamp_min(1).to(losses.scalar_type());
    // Preserve the graph even for a tensor with zero rows or only empty windows.
    return {mean + safe.sum() * 0.0, valid_count};
}

} // namespace cellerator::nn::detail

namespace cellerator::nn {

std::pair<at::Tensor, at::Tensor> reduce_prediction_loss(
    const at::Tensor& losses,
    SupervisionUnit unit,
    std::optional<at::Tensor> valid_mask) {
    if (unit == SupervisionUnit::cell) {
        throw std::invalid_argument(
            "cell reduction requires separate sample/window IDs and physical cell_ids");
    }
    return detail::reduce_prediction_loss_impl(losses, unit, valid_mask, {}, 0);
}

namespace {
std::vector<std::int64_t> validate_group_offsets(
    const at::Tensor& rows,
    const std::vector<std::int64_t>& offsets) {
    if (!rows.defined() || rows.dim() == 0) {
        throw std::invalid_argument("grouped rows must have a leading row axis");
    }
    if (offsets.empty() || offsets.front() != 0 || offsets.back() != rows.size(0)) {
        throw std::invalid_argument("row_offsets must start at zero and end at the row count");
    }
    if (!std::is_sorted(offsets.begin(), offsets.end())) {
        throw std::invalid_argument("row_offsets must be nondecreasing");
    }
    return offsets;
}

at::Tensor grouped_rows(const at::Tensor& rows,
                        const std::vector<std::int64_t>& row_offsets,
                        bool take_last) {
    const auto offsets = validate_group_offsets(rows, row_offsets);
    if (!take_last && !rows.is_floating_point()) {
        throw std::invalid_argument("mean_by_group requires floating-point rows");
    }
    const auto groups = static_cast<std::int64_t>(offsets.size() - 1);
    std::vector<std::int64_t> output_shape = rows.sizes().vec();
    output_shape[0] = groups;
    const auto graph_zero = [&] {
        if (rows.requires_grad() && (rows.is_floating_point() || rows.is_complex())) {
            return at::nan_to_num(rows).sum() * 0.0;
        }
        return at::zeros({}, rows.options());
    };
    if (groups == 0) {
        return at::empty(output_shape, rows.options()) + graph_zero();
    }
    std::vector<at::Tensor> outputs;
    outputs.reserve(static_cast<std::size_t>(groups));
    std::vector<std::int64_t> tail_shape(rows.sizes().begin() + 1, rows.sizes().end());
    for (std::int64_t group = 0; group < groups; ++group) {
        const auto begin = offsets[group];
        const auto end = offsets[group + 1];
        if (begin == end) {
            outputs.push_back(at::zeros(tail_shape, rows.options()));
        } else if (take_last) {
            outputs.push_back(rows.select(0, end - 1));
        } else {
            outputs.push_back(rows.narrow(0, begin, end - begin).mean(0));
        }
    }
    const bool no_rows = rows.size(0) == 0;
    return at::stack(outputs, 0) + (no_rows ? graph_zero() : at::zeros({}, rows.options()));
}
} // namespace

at::Tensor mean_by_group(const at::Tensor& rows,
                         const std::vector<std::int64_t>& row_offsets) {
    return grouped_rows(rows, row_offsets, false);
}

at::Tensor last_by_group(const at::Tensor& rows,
                         const std::vector<std::int64_t>& row_offsets) {
    return grouped_rows(rows, row_offsets, true);
}

} // namespace cellerator::nn
