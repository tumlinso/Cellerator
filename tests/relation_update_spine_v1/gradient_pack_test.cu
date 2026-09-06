#include <Cellerator/compute/architecture/providers/nvidia/sm70/edge_value_gradient/gradient_pack.cuh>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>
using namespace cellerator::compute::architecture::providers::nvidia::sm70::edge_value_gradient;
int main(){
    float *input;__half *output;std::uint32_t *ids;
    assert(cudaMalloc(&input,4*19*sizeof(float))==cudaSuccess);
    assert(cudaMalloc(&output,4*32*sizeof(__half))==cudaSuccess);
    assert(cudaMalloc(&ids,4*sizeof(std::uint32_t))==cudaSuccess);
    const std::uint32_t gather[]={3,1,0xffffffffu,0};
    assert(cudaMemcpy(ids,gather,sizeof(gather),cudaMemcpyHostToDevice)==cudaSuccess);
    const float values[]={0.f,-0.f,1.f+std::ldexp(1.f,-11),1.f+3*std::ldexp(1.f,-11),std::ldexp(1.f,-24),65520.f,
        std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()};
    const unsigned short bits[]={0,0x8000,0x3c00,0x3c02,1,0x7c00,0x7c00};
    std::vector<float> x(4*19);for(unsigned i=0;i<x.size();++i)x[i]=values[i%8];
    assert(cudaMemcpy(input,x.data(),x.size()*sizeof(float),cudaMemcpyHostToDevice)==cudaSuccess);
    pack_request r{};r.input=input;r.input_rows=4;r.input_stride=19;r.input_capacity=x.size();r.output=output;
    r.output_rows=4;r.output_stride=32;r.output_capacity=128;r.gather_ids=ids;r.gather_capacity=4;r.half_rounded=true;
    for(unsigned pass=0;pass<2;++pass){
        if(pass){for(auto& v:x)v=2.f;assert(cudaMemcpy(input,x.data(),x.size()*sizeof(float),cudaMemcpyHostToDevice)==cudaSuccess);}
        assert(cudaMemset(output,0xff,256)==cudaSuccess);
        assert(enqueue_gradient_pack(r)==contract::status_v1::success);
        std::vector<unsigned short> got(128);assert(cudaMemcpy(got.data(),output,256,cudaMemcpyDeviceToHost)==cudaSuccess);
        for(unsigned row=0;row<4;++row)for(unsigned k=0;k<32;++k){auto b=got[row*32+k];
            if(k>=16||gather[row]==0xffffffffu){assert(b==0);continue;}
            if(pass){assert(b==0x4000);continue;}
            auto i=(gather[row]*19+k)%8;if(i==7)assert((b&0x7c00)==0x7c00&&(b&0x3ff));else assert(b==bits[i]);}
    }
    auto bad=r;bad.half_rounded=false;assert(enqueue_gradient_pack(bad)==contract::status_v1::invalid_argument);
    bad=r;bad.output_capacity=127;assert(enqueue_gradient_pack(bad)==contract::status_v1::invalid_argument);
    bad=r;bad.output=output+1;assert(enqueue_gradient_pack(bad)==contract::status_v1::invalid_argument);
    assert(cudaDeviceSynchronize()==cudaSuccess);cudaFree(input);cudaFree(output);cudaFree(ids);
    puts("gradient pack RNE bits/nonfinite/padding/same-address refresh PASS");
}
