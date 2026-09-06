#include <Cellerator/compute/architecture/providers/nvidia/sm70/edge_value_gradient/hybrid_gradient.cuh>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>
using namespace cellerator::compute::architecture::providers::nvidia::sm70::edge_value_gradient;
void fixture(unsigned kind){
    std::vector<contract::edge_ref_v1> edges;
    if(kind<2)for(unsigned s=0;s<16;++s)for(unsigned d=0;d<16;++d)
        if(kind==0||s!=3||d!=5)edges.push_back({s,d,0});
    if(kind)for(unsigned i=0;i<6;++i)edges.push_back({20+i,31-i,0});
    std::reverse(edges.begin(),edges.end());for(unsigned i=0;i<edges.size();++i)edges[i].logical_output_local=i;
    contract::edge_ref_v1 *de;float *x,*y,*out;__half *hx,*hy;
    assert(cudaMalloc(&de,edges.size()*sizeof(edges[0]))==cudaSuccess);
    assert(cudaMemcpy(de,edges.data(),edges.size()*sizeof(edges[0]),cudaMemcpyHostToDevice)==cudaSuccess);
    assert(cudaMalloc(&x,512*sizeof(float))==cudaSuccess);assert(cudaMalloc(&y,512*sizeof(float))==cudaSuccess);
    assert(cudaMalloc(&hx,512*sizeof(__half))==cudaSuccess);assert(cudaMalloc(&hy,512*sizeof(__half))==cudaSuccess);
    assert(cudaMalloc(&out,edges.size()*sizeof(float))==cudaSuccess);
    std::vector<float> a(512),b(512);for(unsigned i=0;i<512;++i){a[i]=(int(i%17)-8)*0.13127f;b[i]=(int(i%11)-5)*0.21731f;}
    contract::support_view_v1 support{de,0,unsigned(edges.size()),32,32};hybrid_gradient *p=nullptr;
    assert(prepare_hybrid_gradient(edges.data(),support,nullptr,0,1<<20,nullptr,&p)==contract::status_v1::success);
    relation_gradient_request r{};r.support=support;r.source=x;r.cotangent=y;r.output=out;
    r.source_capacity=r.cotangent_capacity=512;r.output_capacity=edges.size();r.half_rounded=true;
    r.source_scratch=hx;r.cotangent_scratch=hy;r.source_scratch_capacity=r.cotangent_scratch_capacity=512;
    for(unsigned pass=0;pass<2;++pass){
        if(pass){a[0]=std::numeric_limits<float>::quiet_NaN();b[1]=std::numeric_limits<float>::infinity();}
        assert(cudaMemcpy(x,a.data(),2048,cudaMemcpyHostToDevice)==cudaSuccess);assert(cudaMemcpy(y,b.data(),2048,cudaMemcpyHostToDevice)==cudaSuccess);
        for(bool hybrid:{false,true}){
            assert(cudaMemset(out,0xff,edges.size()*sizeof(float))==cudaSuccess);
            auto result=enqueue_hybrid_gradient(*p,r,hybrid);
            if(kind==2&&hybrid){assert(result==contract::status_v1::unsupported);continue;}
            assert(result==contract::status_v1::success);assert(cudaDeviceSynchronize()==cudaSuccess);
            std::vector<float> got(edges.size());assert(cudaMemcpy(got.data(),out,got.size()*4,cudaMemcpyDeviceToHost)==cudaSuccess);
            for(unsigned i=0;i<edges.size();++i){double ref=0;auto e=edges[i];
                for(unsigned k=0;k<16;++k)ref+=double(__half2float(__float2half_rn(a[e.source_local*16+k])))
                    *double(__half2float(__float2half_rn(b[e.destination_local*16+k])));
                if(std::isnan(ref))assert(std::isnan(got[i]));else if(std::isinf(ref))assert(got[i]==ref);
                else assert(std::fabs(got[i]-ref)<=2e-4+2e-5*std::fabs(ref));}
        }
    }
    auto report=inspect_hybrid_gradient(*p);
    assert(report.sparse_launches==2);
    assert(report.wmma_launches==(kind==2?0:2));assert(report.extraction_launches==report.wmma_launches);
    assert(report.residual_launches==(kind==1?2:0));assert(report.pack_launches==(kind==2?4:12));
    auto strict=r;strict.half_rounded=false;assert(enqueue_hybrid_gradient(*p,strict,true)==contract::status_v1::unsupported);
    auto short_output=r;short_output.output_capacity=0;assert(enqueue_hybrid_gradient(*p,short_output,false)==contract::status_v1::invalid_argument);
    printf("fixture=%u edges=%zu tiles=%llu residual=%llu WMMA=%llu sparse=%llu pack=%llu scratch=%llu PASS\n",kind,edges.size(),
        (unsigned long long)report.tile_count,(unsigned long long)report.residual_count,(unsigned long long)report.wmma_launches,
        (unsigned long long)report.sparse_launches,(unsigned long long)report.pack_launches,(unsigned long long)report.scratch_bytes);
    destroy_hybrid_gradient(p);cudaFree(de);cudaFree(x);cudaFree(y);cudaFree(hx);cudaFree(hy);cudaFree(out);
}
int main(){fixture(0);fixture(1);fixture(2);
    hybrid_gradient *empty=nullptr;assert(prepare_hybrid_gradient(nullptr,{},nullptr,0,1024,nullptr,&empty)==contract::status_v1::success);
    assert(enqueue_hybrid_gradient(*empty,{},false)==contract::status_v1::success);destroy_hybrid_gradient(empty);
    puts("hybrid independent rounded VJP/exact physical writes/nonfinite/empty PASS");}
