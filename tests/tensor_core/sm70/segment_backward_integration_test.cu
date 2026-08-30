#define main ce_geo_segment_normalize_backward_regression
#include "../../relation_algebra/segment_normalize_test.cu"
#undef main

#include "../../../src/compute/architecture/providers/nvidia/sm70/segment_backward_integration.cu"

#include <cassert>

namespace segment = cellerator::compute::segment;
namespace sm70 = cellerator::compute::architecture::providers::nvidia::sm70;

int main() {
    assert(ce_geo_segment_normalize_backward_regression() == 0);
    sm70::prepared_segment_backward_request_v1 missing{};
    assert(sm70::enqueue_prepared_segment_backward_v1(missing)
        == sm70::prepared_segment_backward_status_v1::invalid_argument);
    segment::segment_normalize_plan_v1 invalid_plan{};
    segment::segment_partition_view_v1 partition{};
    missing.plan = &invalid_plan;
    missing.partition = &partition;
    assert(sm70::enqueue_prepared_segment_backward_v1(missing)
        == sm70::prepared_segment_backward_status_v1::invalid_argument);
    return 0;
}
