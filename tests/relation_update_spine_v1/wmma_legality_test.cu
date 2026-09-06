#include <Cellerator/compute/architecture/providers/nvidia/sm70/contract/contract_portfolio_v1.cuh>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
using namespace cellerator::compute::architecture::providers::nvidia::sm70::contract;
int main() {
    __half *a=nullptr,*b=nullptr; float *out=nullptr; rectangular_tile_v1 *dt=nullptr;
    assert(cudaMalloc(&a,32*32*sizeof(__half))==cudaSuccess);
    assert(cudaMalloc(&b,32*32*sizeof(__half))==cudaSuccess);
    assert(cudaMalloc(&out,512*sizeof(float))==cudaSuccess);
    assert(cudaMalloc(&dt,2*sizeof(rectangular_tile_v1))==cudaSuccess);
    std::vector<__half> x(1024); for (unsigned i=0;i<x.size();++i) x[i]=__float2half(float(int(i%13)-6)/8.f);
    assert(cudaMemcpy(a,x.data(),2048,cudaMemcpyHostToDevice)==cudaSuccess);
    assert(cudaMemcpy(b,x.data(),2048,cudaMemcpyHostToDevice)==cudaSuccess);
    rectangular_request_v1 r{}; r.tile_count=1; r.dense={a,b,16}; r.source_count=r.destination_count=32;
    r.projection_output=out;r.source_stride=r.destination_stride=32;
    r.source_capacity=r.destination_capacity=1024;r.output_capacity=512;
    rectangular_tile_v1 t{16,16,256}; prepared_rectangular_v1 p;
    for(unsigned k: {16u,17u,24u}) {
        r.dense.dense_width=k;
        assert(prepare_rectangular_v1(r,&t,dt,2,p)==status_v1::success);
        assert(enqueue_rectangular_mma_residual_v1(p)==status_v1::success);
        assert(cudaDeviceSynchronize()==cudaSuccess);
        std::vector<float> y(512);assert(cudaMemcpy(y.data(),out,2048,cudaMemcpyDeviceToHost)==cudaSuccess);
        for(unsigned i=0;i<16;++i)for(unsigned j=0;j<16;++j){float ref=0;
            for(unsigned c=0;c<k;++c)ref+=__half2float(x[(16+i)*32+c])*__half2float(x[(16+j)*32+c]);
            assert(std::fabs(y[256+i*16+j]-ref)<1e-5f);}
    }
    auto bad=r; bad.source_stride=17;assert(prepare_rectangular_v1(bad,&t,dt,2,p)==status_v1::invalid_argument);
    bad=r;bad.source_stride=24;t.source_begin_local=1;assert(prepare_rectangular_v1(bad,&t,dt,2,p)==status_v1::invalid_argument);t.source_begin_local=16;
    bad=r;bad.source_capacity=1000;assert(prepare_rectangular_v1(bad,&t,dt,2,p)==status_v1::invalid_argument);
    bad=r;bad.dense.source=a+1;assert(prepare_rectangular_v1(bad,&t,dt,2,p)==status_v1::invalid_argument);
    bad=r;bad.output_capacity=511;assert(prepare_rectangular_v1(bad,&t,dt,2,p)==status_v1::invalid_argument);
    bad=r;bad.projection_output=out+1;assert(prepare_rectangular_v1(bad,&t,dt,2,p)==status_v1::invalid_argument);
    bad=r;bad.tile_count=0xffffffffu;assert(prepare_rectangular_v1(bad,&t,dt,0xffffffffu,p)==status_v1::invalid_argument);
    t.source_begin_local=17;assert(prepare_rectangular_v1(r,&t,dt,2,p)==status_v1::invalid_argument);
    bad={};assert(prepare_rectangular_v1(bad,nullptr,nullptr,0,p)==status_v1::success);assert(enqueue_rectangular_mma_residual_v1(p)==status_v1::success);
    assert(enqueue_rectangular_mma_residual_v1(r)==status_v1::unsupported);
    cudaFree(a);cudaFree(b);cudaFree(out);cudaFree(dt);
    puts("WMMA legality actual m16n16k16 + legal tails PASS");
}
