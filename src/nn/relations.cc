#include <Cellerator/nn/relations.hh>

#include <c10/core/InferenceMode.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace cellerator::nn {
namespace {

at::Tensor make_long_tensor(const std::vector<std::int64_t>& values,
                            at::Device device) {
    c10::InferenceMode inference_guard(false);
    auto result = at::tensor(values, at::TensorOptions().dtype(at::kLong));
    return device == at::Device(at::kCPU) ? result : result.to(device);
}

at::Tensor make_bool_tensor(const std::vector<std::uint8_t>& values,
                            at::Device device) {
    c10::InferenceMode inference_guard(false);
    auto result = at::tensor(values, at::TensorOptions().dtype(at::kByte)).to(at::kBool);
    return device == at::Device(at::kCPU) ? result : result.to(device);
}

struct GroupPosition {
    std::int64_t group;
    std::int64_t local;
};

std::vector<GroupPosition> group_positions(
    const biology::OccurrenceOrderToken& order) {
    std::vector<GroupPosition> result(static_cast<std::size_t>(order.extent()));
    const auto& offsets = order.group_offsets();
    for (std::size_t group = 0; group + 1 < offsets.size(); ++group) {
        for (auto position = offsets[group]; position < offsets[group + 1]; ++position) {
            result[static_cast<std::size_t>(position)] = {
                static_cast<std::int64_t>(group), position - offsets[group]};
        }
    }
    return result;
}

void validate_csr_shape(const CsrSupport& support,
                        std::int64_t source_count,
                        std::int64_t target_count) {
    if (target_count == std::numeric_limits<std::int64_t>::max()) {
        throw std::overflow_error("target extent cannot be represented as row_ptr length");
    }
    if (support.row_ptr.size() != static_cast<std::size_t>(target_count + 1)) {
        throw std::invalid_argument("CSR row_ptr needs one entry per target plus terminal offset");
    }
    if (support.col_indices.size() > static_cast<std::size_t>(
            std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error("CSR edge count exceeds int64 range");
    }
    if (support.row_ptr.empty() || support.row_ptr.front() != 0) {
        throw std::invalid_argument("CSR row_ptr must start at zero");
    }
    if (support.row_ptr.back() < 0
        || static_cast<std::uint64_t>(support.row_ptr.back())
            != static_cast<std::uint64_t>(support.col_indices.size())) {
        throw std::invalid_argument("CSR terminal offset must equal the edge count");
    }
    for (std::size_t i = 1; i < support.row_ptr.size(); ++i) {
        if (support.row_ptr[i] < support.row_ptr[i - 1]) {
            throw std::invalid_argument("CSR row_ptr must be nondecreasing");
        }
    }
    for (const auto source : support.col_indices) {
        if (source < 0 || source >= source_count) {
            throw std::invalid_argument("CSR source column is outside the source extent");
        }
    }
}

} // namespace

PreparedRelation PreparedRelation::from_csr(
    CsrSupport support,
    const biology::OccurrenceOrderToken& source_order,
    const biology::OccurrenceOrderToken& target_order,
    std::uint64_t structural_generation,
    at::Device device) {
    return prepare(std::move(support), source_order, target_order,
                   structural_generation, device);
}

PreparedRelation PreparedRelation::prepare(
    CsrSupport support,
    const biology::OccurrenceOrderToken& source_order,
    const biology::OccurrenceOrderToken& target_order,
    std::uint64_t structural_generation,
    at::Device device) {
    if (!source_order.valid() || !target_order.valid()) {
        throw std::invalid_argument("prepared relations require valid endpoint order tokens");
    }
    const auto source_count = source_order.extent();
    const auto target_count = target_order.extent();
    validate_csr_shape(support, source_count, target_count);
    if (source_order.group_offsets().size() != target_order.group_offsets().size()) {
        throw std::invalid_argument(
            "source and target order tokens must declare the same number of groups");
    }
    if (source_order.structure_generation() != target_order.structure_generation()) {
        throw std::invalid_argument(
            "source and target orders must share a structure generation");
    }

    const auto source_positions = group_positions(source_order);
    const auto target_positions = group_positions(target_order);
    std::vector<std::int64_t> rows;
    std::vector<std::uint8_t> causal;
    rows.reserve(support.col_indices.size());
    causal.reserve(support.col_indices.size());
    for (std::int64_t target = 0; target < target_count; ++target) {
        const auto begin = support.row_ptr[static_cast<std::size_t>(target)];
        const auto end = support.row_ptr[static_cast<std::size_t>(target + 1)];
        for (auto edge = begin; edge < end; ++edge) {
            const auto source = support.col_indices[static_cast<std::size_t>(edge)];
            const auto source_position = source_positions[static_cast<std::size_t>(source)];
            const auto target_position = target_positions[static_cast<std::size_t>(target)];
            if (source_position.group != target_position.group) {
                throw std::invalid_argument("CSR support crosses a declared group boundary");
            }
            rows.push_back(target);
            causal.push_back(source_position.local <= target_position.local ? 1u : 0u);
        }
    }

    PreparedRelation result;
    result.n_sources_ = source_count;
    result.n_targets_ = target_count;
    result.structural_generation_ = structural_generation;
    result.device_ = device;
    result.source_order_ = source_order;
    result.target_order_ = target_order;
    result.host_row_ptr_ = std::move(support.row_ptr);
    result.host_col_indices_ = std::move(support.col_indices);
    result.host_rows_ = std::move(rows);
    result.host_causal_mask_ = std::move(causal);
    result.row_ptr_ = make_long_tensor(result.host_row_ptr_, device);
    result.col_indices_ = make_long_tensor(result.host_col_indices_, device);
    result.rows_ = make_long_tensor(result.host_rows_, device);
    result.causal_mask_ = make_bool_tensor(result.host_causal_mask_, device);
    result.row_ptr_version_ = result.row_ptr_._version();
    result.col_indices_version_ = result.col_indices_._version();
    result.rows_version_ = result.rows_._version();
    result.causal_mask_version_ = result.causal_mask_._version();
    return result;
}

void PreparedRelation::validate_storage() const {
    if (!row_ptr_.defined() || !col_indices_.defined() || !rows_.defined()
        || !causal_mask_.defined()) {
        throw std::logic_error("prepared relation storage is undefined");
    }
    if (row_ptr_._version() != row_ptr_version_
        || col_indices_._version() != col_indices_version_
        || rows_._version() != rows_version_
        || causal_mask_._version() != causal_mask_version_) {
        throw std::invalid_argument(
            "prepared relation buffers were mutated; prepare a new relation");
    }
}

bool PreparedRelation::has_future_edges() const {
    validate_storage();
    return std::find(host_causal_mask_.begin(), host_causal_mask_.end(), 0u)
        != host_causal_mask_.end();
}

PreparedRelation PreparedRelation::causal() const {
    validate_storage();
    CsrSupport filtered;
    filtered.row_ptr.reserve(host_row_ptr_.size());
    filtered.row_ptr.push_back(0);
    for (std::int64_t target = 0; target < n_targets_; ++target) {
        const auto begin = host_row_ptr_[static_cast<std::size_t>(target)];
        const auto end = host_row_ptr_[static_cast<std::size_t>(target + 1)];
        for (auto edge = begin; edge < end; ++edge) {
            const auto index = static_cast<std::size_t>(edge);
            if (host_causal_mask_[index]) {
                filtered.col_indices.push_back(host_col_indices_[index]);
            }
        }
        filtered.row_ptr.push_back(static_cast<std::int64_t>(filtered.col_indices.size()));
    }
    return prepare(std::move(filtered), source_order_, target_order_,
                   structural_generation_, device_);
}

PreparedRelation PreparedRelation::to(at::Device device) const {
    validate_storage();
    if (device == device_) return *this;
    return prepare(CsrSupport{host_row_ptr_, host_col_indices_},
                   source_order_, target_order_, structural_generation_, device);
}

void PreparedRelation::validate_for(
    const biology::OccurrenceOrderToken& source_order,
    const biology::OccurrenceOrderToken& target_order,
    std::int64_t source_count,
    std::int64_t target_count,
    at::Device device) const {
    validate_storage();
    validate_for(source_order, target_order, source_count, target_count,
                 structural_generation_, device);
}

void PreparedRelation::validate_for(
    const biology::OccurrenceOrderToken& source_order,
    const biology::OccurrenceOrderToken& target_order,
    std::int64_t source_count,
    std::int64_t target_count,
    std::uint64_t structural_generation,
    at::Device device) const {
    validate_storage();
    if (!source_order.same_order(source_order_)) {
        throw std::invalid_argument("source occurrence order token is stale or incompatible");
    }
    if (!target_order.same_order(target_order_)) {
        throw std::invalid_argument("target occurrence order token is stale or incompatible");
    }
    if (source_count != n_sources_ || source_count != source_order.extent()) {
        throw std::invalid_argument("prepared relation source extent does not match input");
    }
    if (target_count != n_targets_ || target_count != target_order.extent()) {
        throw std::invalid_argument("prepared relation target extent does not match input");
    }
    if (source_order.group_offsets() != source_order_.group_offsets()
        || target_order.group_offsets() != target_order_.group_offsets()) {
        throw std::invalid_argument("prepared relation group offsets differ from endpoints");
    }
    if (source_order.axis() != source_order_.axis()
        || target_order.axis() != target_order_.axis()) {
        throw std::invalid_argument("prepared relation axes differ from endpoints");
    }
    if (source_order.structure_generation() != source_order_.structure_generation()
        || target_order.structure_generation() != target_order_.structure_generation()) {
        throw std::invalid_argument("prepared relation endpoint generation is stale");
    }
    if (structural_generation != structural_generation_) {
        throw std::invalid_argument("prepared relation structural generation is stale");
    }
    if (device != device_) {
        throw std::invalid_argument(
            "prepared relation device differs; call relation.to(device) explicitly");
    }
}

PreparedRelation PreparedRelation::transpose() const {
    validate_storage();
    std::vector<std::vector<std::int64_t>> rows_by_target(
        static_cast<std::size_t>(n_sources_));
    for (std::int64_t target = 0; target < n_targets_; ++target) {
        const auto begin = host_row_ptr_[static_cast<std::size_t>(target)];
        const auto end = host_row_ptr_[static_cast<std::size_t>(target + 1)];
        for (auto edge = begin; edge < end; ++edge) {
            const auto source = host_col_indices_[static_cast<std::size_t>(edge)];
            rows_by_target[static_cast<std::size_t>(source)].push_back(target);
        }
    }
    CsrSupport reversed;
    reversed.row_ptr.reserve(static_cast<std::size_t>(n_sources_ + 1));
    reversed.row_ptr.push_back(0);
    for (const auto& row : rows_by_target) {
        reversed.col_indices.insert(reversed.col_indices.end(), row.begin(), row.end());
        reversed.row_ptr.push_back(static_cast<std::int64_t>(reversed.col_indices.size()));
    }
    return prepare(std::move(reversed), target_order_, source_order_,
                   structural_generation_, device_);
}

} // namespace cellerator::nn
