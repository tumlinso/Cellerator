#include <Cellerator/compiler/profile/represent_value_and_numerical_evidence_v1.hh>

#include <cassert>
#include <cmath>
#include <limits>

int main() {
    using namespace cellerator::compiler::profile::v1;
    const double values[] = {0.0, 1.0, 2.0, 3.0, 4.0,
                             std::numeric_limits<double>::quiet_NaN()};
    const double updates[] = {-0.5, 2.5};
    value_profile_evidence_v1 evidence{};
    assert(summarize_value_profile_evidence_v1(values, 6u, updates, 2u,
               {1u, 2u}, {3u, 4u}, 0.9, &evidence)
           == value_profile_evidence_status_v1::ok);
    assert(evidence.finite_count == 5u && evidence.zero_count == 1u
           && evidence.nonfinite_count == 1u);
    assert(evidence.minimum == 0.0 && evidence.maximum == 4.0
           && evidence.mean == 2.0 && evidence.variance == 2.0);
    assert(evidence.q25 == 1.0 && evidence.median == 2.0 && evidence.q75 == 3.0);
    assert(evidence.maximum_update_magnitude == 2.5 && evidence.dynamic_range == 4.0);
    assert(std::abs(evidence.approximation_risk - 1.0 / 6.0) < 1e-12);
    assert(validate_value_profile_evidence_v1(evidence)
           == value_profile_evidence_status_v1::ok);
}
