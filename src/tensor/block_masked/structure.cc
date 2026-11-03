#include "Cellerator/tensor/block_masked/structure.hh"

#include "Cellerator/tensor/block_masked/view.hh"

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace cellerator::tensor::block_masked {
namespace {

using offset_type = std::uint32_t;

void append_offset(std::vector<offset_type> *offsets, std::uint64_t value) {
    if (value > std::numeric_limits<offset_type>::max()) {
        throw std::overflow_error("block-masked offset exceeds the selected offset type");
    }
    offsets->push_back(static_cast<offset_type>(value));
}

std::uint64_t count_bits(std::uint32_t bits) noexcept {
    std::uint64_t count = 0;
    while (bits != 0) {
        bits &= bits - 1;
        ++count;
    }
    return count;
}

template<class T>
void require_unique(const std::vector<T> &values, const char *description) {
    std::vector<T> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        throw std::invalid_argument(description);
    }
}

void make_identity_map(std::uint64_t count, std::vector<std::uint64_t> *map) {
    if (count > map->max_size()) {
        throw std::length_error("canonical identity map exceeds host vector capacity");
    }
    map->reserve(static_cast<std::size_t>(count));
    for (std::uint64_t coordinate = 0; coordinate < count; ++coordinate) {
        map->push_back(coordinate);
    }
}

} // namespace

Structure Structure::from_rows(
    std::uint64_t features,
    const std::vector<std::vector<std::uint32_t>> &physical_features_by_row,
    std::uint32_t block_width,
    std::uint32_t tile_row_width,
    std::vector<std::uint64_t> canonical_features,
    std::vector<std::uint64_t> canonical_rows) {
    if (block_width == 0 || block_width > 32) {
        throw std::invalid_argument("block_width must be in [1, 32]");
    }
    if (tile_row_width == 0 || tile_row_width > 32) {
        throw std::invalid_argument("tile_row_width must be in [1, 32]");
    }
    if (features > std::numeric_limits<offset_type>::max()) {
        throw std::overflow_error("feature extent exceeds the default uint32 offset range");
    }

    Structure result;
    result.rows_ = static_cast<std::uint64_t>(physical_features_by_row.size());
    result.features_ = features;
    result.tile_row_width_ = tile_row_width;
    result.tiles_ = result.rows_ == 0
        ? 0
        : 1 + (result.rows_ - 1) / tile_row_width;
    result.feature_blocks_ = features == 0
        ? 0
        : 1 + (features - 1) / block_width;

    if (canonical_features.empty()) {
        make_identity_map(features, &canonical_features);
    } else if (canonical_features.size() != features) {
        throw std::invalid_argument("canonical feature map length must equal features");
    }
    if (canonical_rows.empty()) {
        make_identity_map(result.rows_, &canonical_rows);
    } else if (canonical_rows.size() != result.rows_) {
        throw std::invalid_argument("canonical row map length must equal rows");
    }
    require_unique(canonical_rows, "canonical row coordinates must be unique");
    result.canonical_features_ = std::move(canonical_features);
    result.canonical_rows_ = std::move(canonical_rows);

    result.identity_feature_map_ = true;
    for (std::uint64_t physical = 0; physical < features; ++physical) {
        if (result.canonical_features_[static_cast<std::size_t>(physical)] != physical) {
            result.identity_feature_map_ = false;
            break;
        }
    }
    if (!result.identity_feature_map_) {
        result.canonical_to_physical_feature_.reserve(
            static_cast<std::size_t>(features));
        for (std::uint64_t physical = 0; physical < features; ++physical) {
            const auto inserted = result.canonical_to_physical_feature_.emplace(
                result.canonical_features_[static_cast<std::size_t>(physical)],
                static_cast<std::uint32_t>(physical));
            if (!inserted.second) {
                throw std::invalid_argument("canonical feature coordinates must be unique");
            }
        }
    }

    result.feature_block_offsets_.clear();
    if (result.feature_blocks_ >= result.feature_block_offsets_.max_size()) {
        throw std::length_error("feature block offset array exceeds host vector capacity");
    }
    result.feature_block_offsets_.reserve(
        static_cast<std::size_t>(result.feature_blocks_ + 1));
    append_offset(&result.feature_block_offsets_, 0);
    std::uint64_t feature_end = 0;
    for (std::uint64_t block = 0; block < result.feature_blocks_; ++block) {
        const std::uint64_t width = std::min<std::uint64_t>(block_width, features - feature_end);
        feature_end += width;
        append_offset(&result.feature_block_offsets_, feature_end);
    }

    // Normalize each row's declared support. This consumes only coordinate
    // metadata: numerical values, including zero values, never define topology.
    std::vector<std::vector<std::uint32_t>> row_support = physical_features_by_row;
    for (auto &support : row_support) {
        for (std::uint32_t physical : support) {
            if (static_cast<std::uint64_t>(physical) >= features) {
                throw std::out_of_range("physical feature is outside the feature extent");
            }
        }
        std::sort(support.begin(), support.end());
        if (std::adjacent_find(support.begin(), support.end()) != support.end()) {
            throw std::invalid_argument("a row cannot allocate the same feature more than once");
        }
    }

    result.tile_block_offsets_.clear();
    result.tile_block_offsets_.push_back(0);
    result.block_ids_.clear();
    result.participating_row_masks_.clear();
    result.block_row_entry_offsets_.clear();
    result.block_row_entry_offsets_.push_back(0);
    result.feature_masks_.clear();
    result.row_block_value_offsets_.clear();
    result.row_block_value_offsets_.push_back(0);
    std::uint64_t stored_values = 0;

    for (std::uint64_t tile = 0; tile < result.tiles_; ++tile) {
        const std::uint64_t row_begin = tile * tile_row_width;
        const std::uint64_t row_end = row_begin +
            std::min<std::uint64_t>(result.rows_ - row_begin, tile_row_width);

        // The outer ordered map yields the canonical ascending descriptor id;
        // each descriptor's entries are appended in ascending tile-local row.
        std::map<std::uint32_t, std::vector<std::pair<std::uint32_t, std::uint32_t>>> entries;
        for (std::uint64_t row = row_begin; row < row_end; ++row) {
            std::map<std::uint32_t, std::uint32_t> row_masks;
            for (std::uint32_t physical : row_support[static_cast<std::size_t>(row)]) {
                const std::uint32_t block = physical / block_width;
                const std::uint32_t local = physical % block_width;
                row_masks[block] |= (std::uint32_t{1} << local);
            }
            const auto local_row = static_cast<std::uint32_t>(row - row_begin);
            for (const auto &block_mask : row_masks) {
                entries[block_mask.first].emplace_back(local_row, block_mask.second);
            }
        }

        for (const auto &descriptor : entries) {
            const std::uint32_t block_id = descriptor.first;
            std::uint32_t participating_rows = 0;
            for (const auto &row_entry : descriptor.second) {
                participating_rows |= (std::uint32_t{1} << row_entry.first);
            }
            result.block_ids_.push_back(block_id);
            result.participating_row_masks_.push_back(participating_rows);
            for (const auto &row_entry : descriptor.second) {
                result.feature_masks_.push_back(row_entry.second);
                const std::uint64_t value_count = count_bits(row_entry.second);
                if (value_count > std::numeric_limits<std::uint64_t>::max() - stored_values) {
                    throw std::overflow_error("block-masked stored value count overflow");
                }
                stored_values += value_count;
                append_offset(&result.row_block_value_offsets_, stored_values);
            }
            append_offset(
                &result.block_row_entry_offsets_,
                static_cast<std::uint64_t>(result.feature_masks_.size()));
        }
        append_offset(
            &result.tile_block_offsets_,
            static_cast<std::uint64_t>(result.block_ids_.size()));
    }

    result.tile_blocks_ = static_cast<std::uint64_t>(result.block_ids_.size());
    result.row_block_entries_ = static_cast<std::uint64_t>(result.feature_masks_.size());
    result.stored_values_ = stored_values;
    validate(result.view());
    return result;
}

LayoutView<> Structure::view() const & noexcept {
    LayoutView<> result;
    result.tile_block_offsets = tile_block_offsets_.data();
    result.block_ids = block_ids_.data();
    result.participating_row_masks = participating_row_masks_.data();
    result.block_row_entry_offsets = block_row_entry_offsets_.data();
    result.feature_masks = feature_masks_.data();
    result.row_block_value_offsets = row_block_value_offsets_.data();
    result.feature_block_offsets = feature_block_offsets_.data();
    result.tile_row_width = tile_row_width_;
    result.rows = rows_;
    result.features = features_;
    result.tiles = tiles_;
    result.stored_values = stored_values_;
    result.feature_blocks = feature_blocks_;
    result.tile_blocks = tile_blocks_;
    result.row_block_entries = row_block_entries_;
    return result;
}

std::uint32_t Structure::physical_feature(std::uint64_t canonical_coordinate) const {
    if (identity_feature_map_) {
        if (canonical_coordinate < features_) {
            return static_cast<std::uint32_t>(canonical_coordinate);
        }
    } else {
        const auto found = canonical_to_physical_feature_.find(canonical_coordinate);
        if (found != canonical_to_physical_feature_.end()) return found->second;
    }
    throw std::out_of_range("canonical feature coordinate is not bound by this structure");
}

std::uint64_t Structure::value_offset(
    std::uint64_t row,
    std::uint64_t physical_feature) const {
    std::uint64_t offset = 0;
    if (!block_masked::value_offset(view(), row, physical_feature, &offset)) {
        throw std::out_of_range("coordinate is not in the allocated block-masked support");
    }
    return offset;
}

} // namespace cellerator::tensor::block_masked
