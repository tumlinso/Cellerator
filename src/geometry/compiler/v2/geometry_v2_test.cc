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

void test_original_groups_and_incremental_window() {
    const std::uint64_t offsets[] = {0, 2, 4};
    const std::uint64_t items[] = {1, (std::uint64_t{1} << 32) + 1, 3, 4};
    original_group_skeleton skeleton{{11, 12}, offsets, 2, items, 4};
    require(static_cast<bool>(validate_original_group_skeleton(skeleton)));
    const std::uint64_t active[] = {0, 1};
    work_window_change change{1, window_change_kind::add_group, {}};
    incremental_work_window window{{13, 14}, skeleton.identity, {}, active, 2, &change, 1};
    require(static_cast<bool>(validate_incremental_work_window(skeleton, window)));
    const std::uint64_t unsorted[] = {1, 0};
    window.active_original_group_ids = unsorted;
    require(validate_incremental_work_window(skeleton, window).code
        == workload_status_code::invalid_argument);
}
}

int main() { test_workload_profile(); test_original_groups_and_incremental_window(); return 0; }
