#pragma once

#include <Cellerator/layout/occurrence_map.hh>

#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace cellerator::biology {
template<class Identity>
class FeatureBinding;
} // namespace cellerator::biology

namespace cellerator::execution {

// A prepared source-to-model feature mapping.  It is an execution artifact:
// biological identity resolution happens in FeatureBinding, while this value
// carries the integer coordinates consumed by a backend.  Physical positions
// stay associated with their occurrence-order token through OccurrenceMap.
template<class Identity>
class FeatureSourceBinding {
public:
    const std::vector<Identity>& canonical_ids() const noexcept {
        return canonical_ids_;
    }
    const std::vector<Identity>& source_ids() const noexcept {
        return source_ids_;
    }
    const std::vector<std::int64_t>& source_positions() const noexcept {
        return source_positions_;
    }
    const std::vector<std::int64_t>& gather_positions() const noexcept {
        return source_positions_;
    }
    const std::vector<std::int64_t>& output_columns() const noexcept {
        return source_positions_;
    }
    const std::vector<std::optional<std::int64_t>>& canonical_indices() const noexcept {
        return canonical_indices_;
    }
    const std::vector<std::int64_t>& model_indices() const noexcept {
        return model_indices_;
    }
    const std::vector<Identity>& dropped_ids() const noexcept {
        return dropped_ids_;
    }
    const std::optional<layout::OccurrenceMap>& occurrence_map() const noexcept {
        return occurrence_map_;
    }
    const std::vector<std::int64_t>& physical_positions() const noexcept {
        static const std::vector<std::int64_t> empty;
        return occurrence_map_ ? occurrence_map_->physical_positions() : empty;
    }
    bool has_physical_positions() const noexcept {
        return occurrence_map_.has_value();
    }

    // Sparse or externally managed values opt into explicit gathering through
    // their own representation.  The binding never densifies or owns them.
    template<class Values>
        requires requires(Values&& values,
                          const std::vector<std::int64_t>& positions) {
            std::forward<Values>(values).gather_feature_axis(positions, 0);
        }
    auto gather(Values&& values, int axis) const
        -> decltype(std::forward<Values>(values).gather_feature_axis(
            std::declval<const std::vector<std::int64_t>&>(),
            std::declval<int>())) {
        return std::forward<Values>(values).gather_feature_axis(source_positions_, axis);
    }

    template<class Values>
        requires (!requires(Values&& values,
                            const std::vector<std::int64_t>& positions) {
            std::forward<Values>(values).gather_feature_axis(positions, 0);
        } && requires(Values&& values,
                      const std::vector<std::int64_t>& positions) {
            std::forward<Values>(values).gather_feature_columns(positions);
        })
    auto gather(Values&& values, int) const
        -> decltype(std::forward<Values>(values).gather_feature_columns(
            std::declval<const std::vector<std::int64_t>&>())) {
        return std::forward<Values>(values).gather_feature_columns(source_positions_);
    }

private:
    std::vector<Identity> canonical_ids_;
    std::vector<Identity> source_ids_;
    std::vector<std::int64_t> source_positions_;
    std::vector<std::optional<std::int64_t>> canonical_indices_;
    std::vector<std::int64_t> model_indices_;
    std::vector<Identity> dropped_ids_;
    std::optional<layout::OccurrenceMap> occurrence_map_;
    std::shared_ptr<const std::uint8_t> binding_identity_;

    template<class Id>
    friend class ::cellerator::biology::FeatureBinding;
};

} // namespace cellerator::execution
