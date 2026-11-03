#include <Cellerator/modeling/cuda/factorized_interaction.hh>

#include <vector>

int main() {
    using namespace cellerator;

    // Exercise the installed host preparation and workspace query symbols.
    // This binary is only compiled and linked in the package check; running a
    // CUDA workload remains Scheduler-owned.
    std::vector<layout::FactorizedInteractionMechanism> mechanisms;
    const auto prepared = layout::prepare_factorized_interactions(
        mechanisms, layout::FactorizedInteractionExtents{});

    modeling::cuda::FactorizedInteractionWorkspace workspace{};
    return modeling::cuda::factorized_interaction_workspace(
               prepared.view(), {0, 0},
               modeling::cuda::FactorizedInteractionRoute::packet,
               &workspace) == cudaSuccess
        ? 0
        : 1;
}
