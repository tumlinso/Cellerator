#include <Cellerator/models/developmental_time.hh>
#include <Cellerator/models/developmental_time_trajectory.hh>

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

    return (plain_config.input_genes == trajectory_config.input_genes
            && trajectory_batch.features.layout == dt::DevelopmentalTimeLayout::blocked_ell)
        ? 0
        : 1;
}
