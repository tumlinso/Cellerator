#include <Cellerator/examples/models/developmental_time/developmentalTime.hh>
#include <Cellerator/examples/models/developmental_time_trajectory/developmentalTimeTrajectory.hh>
#include "model_runtime_fixtures.cuh"

int main() {
    namespace dt = ::cellerator::models::developmental_time;
    namespace dtt = ::cellerator::models::developmental_time_trajectory;

    dt::DevelopmentalTimeModelConfig plain_config;
    plain_config.input_genes = 8u;
    plain_config.stem_dim = 16u;
    plain_config.hidden_dim = 16u;

    dtt::DevelopmentalTimeTrajectoryModelConfig trajectory_config;
    trajectory_config.input_genes = 8u;
    trajectory_config.stem_dim = 16u;
    trajectory_config.hidden_dim = 16u;

    dt::DevelopmentalTimeBatchView plain_batch{};
    plain_batch.layout = dt::DevelopmentalTimeLayout::blocked_ell;

    dtt::DevelopmentalTimeTrajectoryBatchView trajectory_batch{};
    trajectory_batch.features = plain_batch;
    trajectory_batch.graph.edge_count = 0u;

    const unsigned int row_offsets[] = {0u, 2u, 5u, 6u};
    const unsigned int widths[] = {3u, 1u, 4u};
    const unsigned int expected_slot_offsets[] = {0u, 6u, 9u};
    unsigned int slot_offsets[3] = {};
    model_test::derive_sliced_slot_offsets(3u, row_offsets, widths, slot_offsets);
    model_test::derive_sliced_slot_offsets(0u, nullptr, nullptr, nullptr);
    bool slots_match = true;
    for (std::size_t i = 0u; i < 3u && slots_match; ++i)
        slots_match = slot_offsets[i] == expected_slot_offsets[i];

    return (plain_config.input_genes == trajectory_config.input_genes
            && trajectory_batch.features.layout == dt::DevelopmentalTimeLayout::blocked_ell
            && slots_match)
        ? 0
        : 1;
}
