#pragma once

#include <ATen/ATen.h>
#include <Cellerator/biology/feature_binding.hh>
#include <Cellerator/nn/query.hh>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace cellerator::nn {

enum class SupervisionUnit : std::uint8_t {
    observation,
    cell
};

namespace detail {

template<class Id>
std::vector<std::int64_t> map_groups(
    const std::vector<Id>& prediction_ids,
    const std::vector<Id>& target_ids,
    std::int64_t prediction_groups,
    std::int64_t target_groups,
    const std::optional<std::vector<std::int64_t>>& row_map) {
    if (row_map) {
        if (row_map->size() != static_cast<std::size_t>(prediction_groups)) {
            throw std::invalid_argument("row_map must have one target group per prediction group");
        }
        for (auto group : *row_map) {
            if (group < 0 || group >= target_groups) {
                throw std::out_of_range("row_map names a target group outside the target rows");
            }
        }
        return *row_map;
    }

    if (prediction_ids.empty() && target_ids.empty()) {
        if (prediction_groups != target_groups) {
            throw std::invalid_argument(
                "unlabelled prediction and target group counts differ");
        }
        std::vector<std::int64_t> result(static_cast<std::size_t>(prediction_groups));
        for (std::int64_t i = 0; i < prediction_groups; ++i) result[i] = i;
        return result;
    }
    if (prediction_ids.empty() || target_ids.empty()) {
        throw std::invalid_argument(
            "sample/window IDs must be supplied on both sides or resolved by row_map");
    }
    if (prediction_ids.size() != static_cast<std::size_t>(prediction_groups) ||
        target_ids.size() != static_cast<std::size_t>(target_groups)) {
        throw std::invalid_argument("sample/window IDs must label every coordinate group");
    }
    for (std::size_t i = 0; i < prediction_ids.size(); ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            if (prediction_ids[i] == prediction_ids[j]) {
                throw std::invalid_argument(
                    "ambiguous repeated prediction sample/window ID; use row_map");
            }
        }
    }
    for (std::size_t i = 0; i < target_ids.size(); ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            if (target_ids[i] == target_ids[j]) {
                throw std::invalid_argument(
                    "ambiguous repeated target sample/window ID; use row_map");
            }
        }
    }
    std::vector<std::int64_t> result;
    result.reserve(prediction_ids.size());
    for (const auto& id : prediction_ids) {
        auto found = std::find(target_ids.begin(), target_ids.end(), id);
        if (found == target_ids.end()) {
            throw std::out_of_range("prediction sample/window ID is absent from targets");
        }
        result.push_back(static_cast<std::int64_t>(found - target_ids.begin()));
    }
    return result;
}

inline std::vector<std::int64_t> validate_group_offsets(
    const std::vector<std::int64_t>& offsets, std::int64_t rows, const char* name) {
    if (offsets.empty()) {
        std::vector<std::int64_t> identity(static_cast<std::size_t>(rows + 1));
        for (std::int64_t i = 0; i <= rows; ++i) identity[i] = i;
        return identity;
    }
    if (offsets.front() != 0 || offsets.back() != rows) {
        throw std::invalid_argument(std::string(name) +
            " must start at zero and end at the packed row count");
    }
    if (!std::is_sorted(offsets.begin(), offsets.end())) {
        throw std::invalid_argument(std::string(name) + " must be nondecreasing");
    }
    return offsets;
}

inline std::vector<std::int64_t> row_groups(
    const std::vector<std::int64_t>& offsets, std::int64_t rows) {
    std::vector<std::int64_t> result(static_cast<std::size_t>(rows));
    for (std::size_t group = 0; group + 1 < offsets.size(); ++group) {
        for (std::int64_t row = offsets[group]; row < offsets[group + 1]; ++row) {
            result[static_cast<std::size_t>(row)] = static_cast<std::int64_t>(group);
        }
    }
    return result;
}

std::pair<at::Tensor, at::Tensor> reduce_prediction_loss_impl(
    const at::Tensor& losses, SupervisionUnit unit,
    const std::optional<at::Tensor>& valid_mask,
    const std::vector<std::int64_t>& row_owner_indices,
    std::int64_t owner_count);

} // namespace detail

// Align a tensor-valued Prediction to target feature identities and explicit
// observation/window coordinates. Feature IDs, model indices, and output
// columns remain separate coordinates throughout this operation.
template<class FeatureId, class ObservationId, class Alignment>
std::pair<at::Tensor, at::Tensor> align_prediction_targets(
    const Prediction<at::Tensor, QueryPayload<std::vector<FeatureId>,
                                                ObservationCoordinates<ObservationId>>>& prediction,
    const at::Tensor& target_values,
    const Alignment& prepared_alignment,
    const ObservationCoordinates<ObservationId>& target_coordinates,
    std::optional<std::vector<std::int64_t>> row_map = std::nullopt,
    std::int64_t feature_axis = -1) {
    if (!target_values.defined() || !target_values.is_floating_point()) {
        throw std::invalid_argument("target_values must be a defined floating-point tensor");
    }
    if (!prediction.values.defined() || !prediction.values.is_floating_point()) {
        throw std::invalid_argument("prediction values must be a defined floating-point tensor");
    }
    const auto& predicted_ids = prediction.payload.feature_ids;
    if (predicted_ids != prepared_alignment.canonical_ids()) {
        throw std::invalid_argument(
            "prepared feature destination order must match Prediction feature IDs");
    }
    if (prediction.values.dim() < 2 || target_values.dim() != prediction.values.dim()) {
        throw std::invalid_argument(
            "prediction and target tensors must have matching rank and observation/feature axes");
    }
    auto rank = static_cast<std::int64_t>(target_values.dim());
    auto axis = feature_axis < 0 ? feature_axis + rank : feature_axis;
    if (axis <= 0 || axis >= rank) {
        throw std::invalid_argument("feature_axis must identify a non-leading tensor axis");
    }
    if (target_values.size(axis) !=
        static_cast<std::int64_t>(prepared_alignment.source_ids().size())) {
        throw std::invalid_argument(
            "target feature axis does not match the prepared source feature order");
    }
    if (prediction.values.size(axis) !=
        static_cast<std::int64_t>(predicted_ids.size())) {
        throw std::invalid_argument(
            "prediction feature axis does not match Prediction feature IDs");
    }

    auto target_columns = at::tensor(prepared_alignment.output_columns(),
        at::TensorOptions().dtype(at::kLong).device(target_values.device()));
    auto aligned_targets = target_values.index_select(axis, target_columns);
    const auto prediction_rows = prediction.values.size(0);
    const auto target_rows = aligned_targets.size(0);
    const auto prediction_offsets = detail::validate_group_offsets(
        prediction.payload.coordinates.row_offsets, prediction_rows,
        "prediction row_offsets");
    const auto target_offsets = detail::validate_group_offsets(
        target_coordinates.row_offsets, target_rows, "target row_offsets");
    const auto prediction_groups = static_cast<std::int64_t>(prediction_offsets.size() - 1);
    const auto target_groups = static_cast<std::int64_t>(target_offsets.size() - 1);
    const auto group_mapping = detail::map_groups(
        prediction.payload.coordinates.sample_ids, target_coordinates.sample_ids,
        prediction_groups, target_groups, row_map);

    std::vector<std::int64_t> row_positions;
    row_positions.reserve(static_cast<std::size_t>(prediction_rows));
    for (std::int64_t group = 0; group < prediction_groups; ++group) {
        const auto pstart = prediction_offsets[group];
        const auto pend = prediction_offsets[group + 1];
        const auto tstart = target_offsets[group_mapping[group]];
        const auto tend = target_offsets[group_mapping[group] + 1];
        if (pend - pstart != tend - tstart) {
            throw std::invalid_argument(
                "matched prediction and target groups have different row counts");
        }
        for (auto row = tstart; row < tend; ++row) row_positions.push_back(row);
    }
    if (row_positions.size() != static_cast<std::size_t>(prediction_rows)) {
        throw std::invalid_argument("row alignment did not cover every prediction row");
    }
    auto row_index = at::tensor(row_positions,
        at::TensorOptions().dtype(at::kLong).device(aligned_targets.device()));
    aligned_targets = aligned_targets.index_select(0, row_index);
    if (prediction.values.sizes() != aligned_targets.sizes()) {
        throw std::invalid_argument(
            "aligned prediction and target tensor shapes must match exactly");
    }
    return {prediction.values, std::move(aligned_targets)};
}

template<class FeatureId, class ObservationId, class Label>
at::Tensor categorical_target_columns(
    const Prediction<at::Tensor, QueryPayload<std::vector<FeatureId>,
                                                ObservationCoordinates<ObservationId>>>& prediction,
    const std::vector<Label>& event_labels) {
    std::vector<std::int64_t> columns;
    columns.reserve(event_labels.size());
    const auto& feature_ids = prediction.payload.feature_ids;
    for (std::size_t i = 0; i < feature_ids.size(); ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            if (feature_ids[i] == feature_ids[j]) {
                throw std::invalid_argument("Prediction has repeated class feature IDs");
            }
        }
    }
    for (const auto& label : event_labels) {
        auto found = std::find(feature_ids.begin(), feature_ids.end(), label);
        if (found == feature_ids.end()) {
            throw std::out_of_range("event label has no Prediction class column");
        }
        columns.push_back(static_cast<std::int64_t>(found - feature_ids.begin()));
    }
    return at::tensor(columns,
        at::TensorOptions().dtype(at::kLong).device(prediction.values.device()));
}

// The unit is mandatory. Cell reduction pools every valid scalar term across
// all windows of each physical cell, then averages valid cells equally.
template<class ObservationId, class CellId>
std::pair<at::Tensor, at::Tensor> reduce_prediction_loss(
    const at::Tensor& losses,
    SupervisionUnit unit,
    const ObservationCoordinates<ObservationId>& coordinates,
    const std::vector<CellId>& cell_ids,
    std::optional<at::Tensor> valid_mask = std::nullopt) {
    const auto rows = losses.dim() == 0 ? std::int64_t{1} : losses.size(0);
    const auto offsets = detail::validate_group_offsets(coordinates.row_offsets,
                                                         rows, "row_offsets");
    const auto group_count = static_cast<std::int64_t>(offsets.size() - 1);
    if (!coordinates.sample_ids.empty() &&
        coordinates.sample_ids.size() != static_cast<std::size_t>(group_count)) {
        throw std::invalid_argument("sample_ids must label every observation/window group");
    }
    if (unit == SupervisionUnit::observation) {
        if (!cell_ids.empty()) {
            throw std::invalid_argument("cell_ids are only used for cell reduction");
        }
        return detail::reduce_prediction_loss_impl(
            losses, unit, valid_mask, {}, 0);
    }
    if (unit != SupervisionUnit::cell) {
        throw std::invalid_argument("unit must be SupervisionUnit::observation or cell");
    }
    if (coordinates.sample_ids.size() != static_cast<std::size_t>(group_count) ||
        cell_ids.size() != static_cast<std::size_t>(group_count)) {
        throw std::invalid_argument(
            "cell reduction requires separate sample/window IDs and physical cell_ids");
    }
    for (std::size_t i = 0; i < coordinates.sample_ids.size(); ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            if (coordinates.sample_ids[i] == coordinates.sample_ids[j]) {
                throw std::invalid_argument("sample/window IDs must be unique for cell reduction");
            }
        }
    }
    std::vector<CellId> owners;
    std::vector<std::int64_t> group_owner;
    group_owner.reserve(cell_ids.size());
    for (const auto& cell_id : cell_ids) {
        auto found = std::find(owners.begin(), owners.end(), cell_id);
        if (found == owners.end()) {
            owners.push_back(cell_id);
            group_owner.push_back(static_cast<std::int64_t>(owners.size() - 1));
        } else {
            group_owner.push_back(static_cast<std::int64_t>(found - owners.begin()));
        }
    }
    auto row_group = detail::row_groups(offsets, rows);
    std::vector<std::int64_t> row_owner;
    row_owner.reserve(row_group.size());
    for (auto group : row_group) row_owner.push_back(group_owner[group]);
    return detail::reduce_prediction_loss_impl(
        losses, unit, valid_mask, row_owner, static_cast<std::int64_t>(owners.size()));
}

template<class ObservationId>
std::pair<at::Tensor, at::Tensor> reduce_prediction_loss(
    const at::Tensor& losses,
    SupervisionUnit unit,
    const ObservationCoordinates<ObservationId>& coordinates,
    std::optional<at::Tensor> valid_mask = std::nullopt) {
    if (unit == SupervisionUnit::cell) {
        throw std::invalid_argument(
            "cell reduction requires separate sample/window IDs and physical cell_ids");
    }
    const auto rows = losses.dim() == 0 ? std::int64_t{1} : losses.size(0);
    const auto offsets = detail::validate_group_offsets(coordinates.row_offsets,
                                                         rows, "row_offsets");
    const auto group_count = static_cast<std::int64_t>(offsets.size() - 1);
    if (!coordinates.sample_ids.empty() &&
        coordinates.sample_ids.size() != static_cast<std::size_t>(group_count)) {
        throw std::invalid_argument("sample_ids must label every observation/window group");
    }
    return detail::reduce_prediction_loss_impl(losses, unit, valid_mask, {}, 0);
}

// Observation-level reduction needs no biological keys, but still requires an
// explicit unit so callers cannot inherit an accidental weighting policy.
std::pair<at::Tensor, at::Tensor> reduce_prediction_loss(
    const at::Tensor& losses,
    SupervisionUnit unit,
    std::optional<at::Tensor> valid_mask = std::nullopt);

// General row grouping utilities used by native model inputs. Offsets are
// explicit host boundaries; empty groups produce zero rows in both reductions.
at::Tensor mean_by_group(const at::Tensor& rows,
                         const std::vector<std::int64_t>& row_offsets);
at::Tensor last_by_group(const at::Tensor& rows,
                         const std::vector<std::int64_t>& row_offsets);

} // namespace cellerator::nn
