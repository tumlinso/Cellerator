#include <Cellerator/compiler/profile/represent_value_and_numerical_evidence_v1.hh>

#include <cmath>
#include <limits>

namespace cellerator::compiler::profile::v1 {
namespace {
double select_finite(const double *values, std::uint64_t count,
                     std::uint64_t rank) noexcept {
    double selected = 0.0;
    double floor = -std::numeric_limits<double>::infinity();
    for (std::uint64_t k = 0; k <= rank; ++k) {
        double next = std::numeric_limits<double>::infinity();
        for (std::uint64_t i = 0; i < count; ++i)
            if (std::isfinite(values[i]) && values[i] > floor && values[i] < next)
                next = values[i];
        if (!std::isfinite(next)) {
            std::uint64_t equal_count = 0u;
            for (std::uint64_t i = 0; i < count; ++i)
                if (std::isfinite(values[i]) && values[i] == floor) ++equal_count;
            if (k + equal_count > rank) return floor;
            k += equal_count - 1u;
            continue;
        }
        selected = next;
        floor = next;
        std::uint64_t equal_count = 0u;
        for (std::uint64_t i = 0; i < count; ++i)
            if (values[i] == next) ++equal_count;
        if (k + equal_count > rank) return next;
        k += equal_count - 1u;
    }
    return selected;
}
}  // namespace

value_profile_evidence_status_v1 summarize_value_profile_evidence_v1(
    const double *values, std::uint64_t value_count,
    const double *updates, std::uint64_t update_count,
    profile_identity_v1 evidence_identity,
    profile_identity_v1 value_plane_identity,
    double confidence, value_profile_evidence_v1 *evidence) noexcept {
    if (evidence == nullptr || (value_count != 0u && values == nullptr)
        || (update_count != 0u && updates == nullptr))
        return value_profile_evidence_status_v1::invalid_argument;
    if (!std::isfinite(confidence) || confidence < 0.0 || confidence > 1.0)
        return value_profile_evidence_status_v1::invalid_confidence;
    value_profile_evidence_v1 result{};
    result.evidence = evidence_identity;
    result.value_plane = value_plane_identity;
    result.observation_count = value_count;
    result.minimum = std::numeric_limits<double>::infinity();
    result.maximum = -std::numeric_limits<double>::infinity();
    double m2 = 0.0;
    double minimum_nonzero = std::numeric_limits<double>::infinity();
    double maximum_absolute = 0.0;
    for (std::uint64_t i = 0; i < value_count; ++i) {
        const auto value = values[i];
        if (!std::isfinite(value)) { ++result.nonfinite_count; continue; }
        ++result.finite_count;
        if (value == 0.0) ++result.zero_count;
        result.minimum = value < result.minimum ? value : result.minimum;
        result.maximum = value > result.maximum ? value : result.maximum;
        const auto delta = value - result.mean;
        result.mean += delta / static_cast<double>(result.finite_count);
        m2 += delta * (value - result.mean);
        const auto absolute = std::abs(value);
        if (absolute != 0.0 && absolute < minimum_nonzero) minimum_nonzero = absolute;
        if (absolute > maximum_absolute) maximum_absolute = absolute;
    }
    if (result.finite_count == 0u)
        return value_profile_evidence_status_v1::no_finite_values;
    result.variance = m2 / result.finite_count;
    result.q25 = select_finite(values, value_count, (result.finite_count - 1u) / 4u);
    result.median = select_finite(values, value_count, (result.finite_count - 1u) / 2u);
    result.q75 = select_finite(values, value_count,
                               (3u * (result.finite_count - 1u)) / 4u);
    for (std::uint64_t i = 0; i < update_count; ++i)
        if (std::isfinite(updates[i])
            && std::abs(updates[i]) > result.maximum_update_magnitude)
            result.maximum_update_magnitude = std::abs(updates[i]);
    result.dynamic_range = std::isfinite(minimum_nonzero)
        ? maximum_absolute / minimum_nonzero : 0.0;
    result.approximation_risk = static_cast<double>(result.nonfinite_count)
        / (value_count == 0u ? 1.0 : static_cast<double>(value_count));
    if (result.dynamic_range > 65504.0) result.approximation_risk = 1.0;
    result.confidence = confidence;
    *evidence = result;
    return value_profile_evidence_status_v1::ok;
}

value_profile_evidence_status_v1 validate_value_profile_evidence_v1(
    const value_profile_evidence_v1 &evidence) noexcept {
    if (evidence.schema_version != value_profile_evidence_schema_version_v1)
        return value_profile_evidence_status_v1::unsupported_schema;
    if (!std::isfinite(evidence.confidence) || evidence.confidence < 0.0
        || evidence.confidence > 1.0)
        return value_profile_evidence_status_v1::invalid_confidence;
    return evidence.finite_count == 0u
        ? value_profile_evidence_status_v1::no_finite_values
        : value_profile_evidence_status_v1::ok;
}
}  // namespace cellerator::compiler::profile::v1
