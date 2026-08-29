#pragma once

#include <Cellerator/compute/neighbors/forward_neighbors/fn_query.hh>

#include <cuda_runtime_api.h>

namespace cellerator::compute::neighbors::forward_neighbors {

// This is the reusable mathematical boundary. Durable index construction,
// cell lookup, sharding/residency, storage, workflow policy, and owning result
// objects belong to downstream CellShard/BioPrep/application layers.

void initialize_forward_neighbor_result(
    const ForwardNeighborResultDeviceView &result,
    std::int64_t query_rows,
    int top_k,
    cudaStream_t stream = nullptr);

void select_forward_neighbor_ann_lists(
    const ForwardNeighborQueryDeviceView &query,
    const ForwardNeighborAnnListDeviceView &lists,
    const ForwardNeighborAnnSearchConfig &config,
    const ForwardNeighborAnnWorkspaceDeviceView &workspace,
    cudaStream_t stream = nullptr);

void refine_forward_neighbors_dense(
    const ForwardNeighborQueryDeviceView &query,
    const ForwardNeighborDenseIndexDeviceView &index,
    const ForwardNeighborAnnListDeviceView &lists,
    const ForwardNeighborAnnSearchConfig &config,
    const ForwardNeighborAnnWorkspaceDeviceView &workspace,
    const ForwardNeighborResultDeviceView &result,
    cudaStream_t stream = nullptr);

void refine_forward_neighbors_blocked_ell(
    const ForwardNeighborQueryDeviceView &query,
    const ForwardNeighborBlockedEllIndexDeviceView &index,
    const ForwardNeighborAnnListDeviceView &lists,
    const ForwardNeighborAnnSearchConfig &config,
    const ForwardNeighborAnnWorkspaceDeviceView &workspace,
    const ForwardNeighborResultDeviceView &result,
    cudaStream_t stream = nullptr);

void refine_forward_neighbors_sliced_ell(
    const ForwardNeighborQueryDeviceView &query,
    const ForwardNeighborSlicedEllIndexDeviceView &index,
    const ForwardNeighborAnnListDeviceView &lists,
    const ForwardNeighborAnnSearchConfig &config,
    const ForwardNeighborAnnWorkspaceDeviceView &workspace,
    const ForwardNeighborResultDeviceView &result,
    cudaStream_t stream = nullptr);

} // namespace cellerator::compute::neighbors::forward_neighbors
