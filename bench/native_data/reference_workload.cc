#include "workload.hh"

#include "../../tests/native_data/reference/fixture.hh"

namespace cellerator::bench::native_data {
namespace {

using namespace cellerator::test::native_data_reference;

Workload convert_reference(const values& input, std::string_view name) {
    Workload result;
    result.name = name;
    result.coordinates = input.x.size();
    result.conditions = 1;
    result.parameters = 1;
    result.baseline_scale = static_cast<float>(-input.beta);
    result.x.reserve(input.x.size());
    for (const double value : input.x) result.x.push_back(static_cast<float>(value));
    result.z = {static_cast<float>(input.z0), static_cast<float>(input.z1)};
    result.theta.push_back(static_cast<float>(input.theta));
    result.phi.push_back(static_cast<float>(input.phi));
    result.tau.push_back(static_cast<float>(input.tau));
    result.upstream.reserve(loss_seed.size());
    for (const double value : loss_seed) result.upstream.push_back(static_cast<float>(value));

    for (const auto& group : mechanisms) {
        result.argument_begin.push_back(result.argument_coordinate.size());
        result.argument_count.push_back(group.arity);
        for (std::size_t slot = 0; slot < group.arity; ++slot)
            result.argument_coordinate.push_back(group.ordered_inputs[slot]);
        result.output_begin.push_back(result.output_coordinate.size());
        std::uint64_t count = 0;
        for (std::size_t output = 0; output < group.output_scales.size(); ++output) {
            if (group.output_scales[output] == 0.0) continue;
            result.output_coordinate.push_back(output);
            result.output_scale.push_back(static_cast<float>(group.output_scales[output]));
            ++count;
        }
        result.output_count.push_back(count);
        result.parameter_index.push_back(group.parameter_index);
        result.condition_index.push_back(group.condition_index);
        result.group_owner.push_back(group.device);
    }
    return result;
}

} // namespace

Workload make_reference_active() { return convert_reference(active_values, "reference_active"); }
Workload make_reference_inactive() { return convert_reference(inactive_values, "reference_inactive"); }

} // namespace cellerator::bench::native_data
