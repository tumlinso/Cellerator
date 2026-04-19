#include <Cellerator/trajectory/trajectory_tree.cuh>

int main() {
    using namespace cellerator::compute::graph;
    using namespace cellerator::trajectory;

    TrajectoryRecordTable records;
    records.reserve(2, 2);
    records.append(10, 0, 0.10f, {0.8f, 0.6f});
    records.append(11, 0, 0.20f, {0.7f, 0.7f});
    const auto slabs = build_time_slabs(records, 1);
    const auto windows = build_future_window_bounds(records, 0.01f, 0.20f, 1.0e-6f);
    const auto assignments = assign_rows_to_delta_slabs(slabs, records, 0.02f);
    IncrementalInsertPlan plan = plan_incremental_insert(slabs, records, 0.02f);
    (void) windows;
    (void) assignments;
    (void) plan;

    return slabs.size() == 2u ? 0 : 1;
}
