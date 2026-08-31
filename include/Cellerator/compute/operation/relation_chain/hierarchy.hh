#pragma once

#include "Cellerator/compute/operation/relation_chain/plan.hh"

#include <cstddef>
#include <cstdint>

namespace cellerator::compute::relation_chain {

enum class hierarchy_pool_kind : std::uint8_t {
    sum,
    mean
};

// parent_offsets partitions child order. Pool and broadcast are direct stages;
// no adjacency matrix is built or owned by this interface.
inline void hierarchy_pool(const local_index_type* parent_offsets,
                           local_index_type parent_count,
                           std::uint32_t width,
                           hierarchy_pool_kind kind,
                           const float* child_values,
                           float* parent_values) noexcept {
    for (local_index_type parent = 0; parent < parent_count; ++parent) {
        const local_index_type begin = parent_offsets[parent];
        const local_index_type end = parent_offsets[parent + 1];
        for (std::uint32_t feature = 0; feature < width; ++feature) {
            float value = 0.0F;
            for (local_index_type child = begin; child < end; ++child) {
                value += child_values[static_cast<std::size_t>(child) * width + feature];
            }
            if (kind == hierarchy_pool_kind::mean && end != begin) {
                value /= static_cast<float>(end - begin);
            }
            parent_values[static_cast<std::size_t>(parent) * width + feature] = value;
        }
    }
}

inline void hierarchy_broadcast(const local_index_type* parent_offsets,
                                local_index_type parent_count,
                                std::uint32_t width,
                                const float* parent_values,
                                float* child_values) noexcept {
    for (local_index_type parent = 0; parent < parent_count; ++parent) {
        for (local_index_type child = parent_offsets[parent];
             child < parent_offsets[parent + 1]; ++child) {
            for (std::uint32_t feature = 0; feature < width; ++feature) {
                child_values[static_cast<std::size_t>(child) * width + feature] =
                    parent_values[static_cast<std::size_t>(parent) * width + feature];
            }
        }
    }
}

}  // namespace cellerator::compute::relation_chain
