#include <Cellerator/compiler/ir/semantic/implement_contraction_segment_and_normalization_operatio_v1.hh>

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>

using namespace Cellerator::compiler::ir::semantic;

int main() {
    aggregate_operation_definition_ir_v1 contraction;
    contraction.identity = {1, 2};
    contraction.operation = aggregate_operation_ir_v1::support_contraction;
    contraction.support_identity = {3, 4};
    double contracted = 0.0;
    const std::vector<double> left{1, 2, 3, 4};
    const std::vector<double> right{4, 3, 2, 1};
    const std::vector<std::uint8_t> active{1, 0, 1, 0};
    assert(interpret_support_contraction_ir_v1(
        contraction, left, right, active, &contracted) ==
        aggregate_operation_status_ir_v1::success);
    const double reference_contraction = left[0] * right[0] + left[2] * right[2];
    assert(contracted == reference_contraction);

    aggregate_operation_definition_ir_v1 segmented;
    segmented.identity = {5, 6};
    segmented.segment_identity = {7, 8};
    const std::vector<double> values{1, 2, 3, 4};
    const std::vector<std::uint64_t> offsets{0, 2, 4};
    std::vector<double> output;
    segmented.operation = aggregate_operation_ir_v1::segment_sum;
    assert(interpret_segment_operation_ir_v1(segmented, values, offsets, &output) ==
           aggregate_operation_status_ir_v1::success);
    assert((output == std::vector<double>{3, 7}));

    segmented.operation = aggregate_operation_ir_v1::segment_maximum;
    segmented.neutral_element = -std::numeric_limits<double>::infinity();
    assert(interpret_segment_operation_ir_v1(segmented, values, offsets, &output) ==
           aggregate_operation_status_ir_v1::success);
    assert((output == std::vector<double>{2, 4}));

    segmented.operation = aggregate_operation_ir_v1::normalize_softmax;
    segmented.neutral_element = 0.0;
    assert(interpret_segment_operation_ir_v1(segmented, values, offsets, &output) ==
           aggregate_operation_status_ir_v1::success);
    const auto first_segment_sum = output[0] + output[1];
    const auto second_segment_sum = output[2] + output[3];
    assert(std::abs(first_segment_sum - 1.0) < 1e-12);
    assert(std::abs(second_segment_sum - 1.0) < 1e-12);
    assert(lower_segment_operation_ir_v1(segmented.operation) ==
           cellerator::compute::operation::v2::segment_operation::softmax);

    std::cout << "contraction=" << contracted
              << " segments=2 softmax_reference=matched\n";
}
