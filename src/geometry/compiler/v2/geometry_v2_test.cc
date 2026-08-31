#include <Cellerator/geometry/compiler/workload_profile_v2.hh>

#include <cstdlib>

using namespace cellerator::geometry::compiler::v2;

namespace {
void require(bool condition) { if (!condition) std::abort(); }

void test_workload_profile() {
    workload_component component{};
    component.identity = {1, 2};
    component.dense_width_min = 16;
    component.dense_width_max = 128;
    component.dense_width_bucket = 64;
    component.frequency = (std::uint64_t{1} << 32) + 9;
    workload_profile profile{workload_profile_schema_version,
        sizeof(workload_profile), &component, 1};
    require(static_cast<bool>(validate_workload_profile(profile)));
    component.requirement_flags = canonical_output_required | packed_output_permitted;
    require(validate_workload_profile(profile).code
        == workload_status_code::invalid_requirements);
}
}

int main() { test_workload_profile(); return 0; }
