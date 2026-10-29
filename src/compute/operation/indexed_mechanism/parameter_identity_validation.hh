#pragma once

#include <Cellerator/compute/operation/indexed_mechanism/training.hh>

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

namespace cellerator::compute::operation::indexed::detail {

enum class parameter_input_validation {
    valid,
    invalid_identity_or_value,
    duplicate_identity,
};

// Preserve constructor error ordering: an invalid identity/value is reported
// at its original position unless a duplicate occurs earlier in the prefix.
// Sorting a temporary identity list keeps the canonical owner vector intact.
inline parameter_input_validation validate_parameter_inputs(
    std::span<const identity> ids, std::span<const float> values) {
    if (ids.size() != values.size())
        return parameter_input_validation::invalid_identity_or_value;

    std::size_t valid_prefix = ids.size();
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (!v2::valid_stable_id(ids[i]) || !std::isfinite(values[i])) {
            valid_prefix = i;
            break;
        }
    }

    std::vector<identity> sorted(ids.begin(), ids.begin() + valid_prefix);
    std::sort(sorted.begin(), sorted.end(), [](identity left, identity right) {
        return left.high < right.high
            || (left.high == right.high && left.low < right.low);
    });
    for (std::size_t i = 1; i < sorted.size(); ++i)
        if (v2::same_stable_id(sorted[i - 1], sorted[i]))
            return parameter_input_validation::duplicate_identity;

    return valid_prefix == ids.size()
        ? parameter_input_validation::valid
        : parameter_input_validation::invalid_identity_or_value;
}

} // namespace cellerator::compute::operation::indexed::detail
