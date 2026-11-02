#pragma once

#include <Cellerator/structure/coordinates.hh>
#include <stdexcept>

namespace cellerator::layout {

// Physical/partition identity is explicitly associated with an existing
// logical axis; it does not redefine that axis or own its values.
struct AxisBinding {
    structure::LogicalAxis logical{};
    structure::geometry_id geometry{};
    structure::partition_id partition{};

    void validate() const {
        if (!structure::valid_logical_axis(logical)
            || !structure::valid_identity(geometry)
            || !structure::valid_identity(partition))
            throw std::invalid_argument("axis binding identities must be valid");
    }
};

inline bool operator==(const AxisBinding& left, const AxisBinding& right) noexcept {
    return left.logical == right.logical
        && structure::same_identity(left.geometry, right.geometry)
        && structure::same_identity(left.partition, right.partition);
}
inline bool operator!=(const AxisBinding& left, const AxisBinding& right) noexcept {
    return !(left == right);
}

} // namespace cellerator::layout
