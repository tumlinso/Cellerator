#include "../../../src/compute/architecture/providers/nvidia/sm70/exchange_cover_native_normalize.cu"

#include <cuda_runtime.h>
#include <cassert>
#include <cmath>

namespace projection = cellerator::compute::projection;
namespace sm70 = cellerator::compute::architecture::providers::nvidia::sm70;
namespace { void ok(cudaError_t status) { assert(status == cudaSuccess); } }

int main() {
    sm70::support_projection_edge_v1 edges[4]{};
    edges[0].region_kind = projection::physical_region_kind_v1::mma;
    edges[0].stable_output_index = 3u;
    edges[1].region_kind = projection::physical_region_kind_v1::mma;
    edges[1].stable_output_index = 0u;
    edges[2].region_kind = projection::physical_region_kind_v1::residual;
    edges[2].stable_output_index = 4u;
    edges[3].region_kind = projection::physical_region_kind_v1::residual;
    edges[3].stable_output_index = 1u;
    const sm70::cover_native_partition_v1 partitions[] = {
        {projection::physical_region_kind_v1::mma, 0u, 2u},
        {projection::physical_region_kind_v1::residual, 2u, 2u}};
    const float values[] = {2.0f, -1.0f, 99.0f, 1.0f, 3.0f};
    sm70::support_projection_edge_v1 *de = nullptr;
    sm70::cover_native_partition_v1 *dp = nullptr;
    float *dv = nullptr, *dout = nullptr;
    ok(cudaMalloc(&de, sizeof(edges))); ok(cudaMalloc(&dp, sizeof(partitions)));
    ok(cudaMalloc(&dv, sizeof(values))); ok(cudaMalloc(&dout, sizeof(values)));
    ok(cudaMemcpy(de, edges, sizeof(edges), cudaMemcpyHostToDevice));
    ok(cudaMemcpy(dp, partitions, sizeof(partitions), cudaMemcpyHostToDevice));
    ok(cudaMemcpy(dv, values, sizeof(values), cudaMemcpyHostToDevice));
    sm70::cover_native_normalize_request_v1 request{};
    request.selected_edges=de; request.selected_edge_count=4u;
    request.partitions=dp; request.partition_count=2u;
    request.logical_edge_values=dv; request.logical_edge_count=5u;
    request.logical_edge_output=dout;
    assert(sm70::cover_native_normalize_empirical_required_v1);
    assert(sm70::enqueue_cover_native_normalize_v1(request)
        == sm70::cover_native_normalize_status_v1::success);
    float output[5]{};
    ok(cudaMemcpy(output,dout,sizeof(output),cudaMemcpyDeviceToHost));
    const float mma_sum=std::exp(values[3])+std::exp(values[0]);
    const float residual_sum=std::exp(values[4])+std::exp(values[1]);
    assert(std::fabs(output[3]-std::exp(values[3])/mma_sum)<1e-6f);
    assert(std::fabs(output[0]-std::exp(values[0])/mma_sum)<1e-6f);
    assert(std::fabs(output[4]-std::exp(values[4])/residual_sum)<1e-6f);
    assert(std::fabs(output[1]-std::exp(values[1])/residual_sum)<1e-6f);
    assert(output[2]==0.0f);
    ok(cudaFree(dout)); ok(cudaFree(dv)); ok(cudaFree(dp)); ok(cudaFree(de));
    return 0;
}
