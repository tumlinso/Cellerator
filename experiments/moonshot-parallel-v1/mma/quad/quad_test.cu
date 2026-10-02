#include "quad_mma.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace cellerator::experimental::moonshot;
static void require(bool ok,const char* message) { if(!ok) { std::fprintf(stderr,"FAIL %s\n",message); std::exit(1); } }
static void check(cudaError_t e) { if(e!=cudaSuccess) { std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); std::exit(1); } }
static void host_checks() {
    const auto a=reinterpret_cast<const __half*>(0x1000), b=reinterpret_cast<const __half*>(0x2000);
    auto y=reinterpret_cast<float*>(0x3000);
    QuadMmaRequest q{a,b,y,32,32,64,1};
    require(validate_quad_mma(q)==cudaSuccess,"valid metadata");
    q.y_elements=63; require(validate_quad_mma(q)==cudaErrorInvalidValue,"capacity"); q.y_elements=64;
    q.y=reinterpret_cast<float*>(0x1000); require(validate_quad_mma(q)==cudaErrorInvalidValue,"alias"); q.y=y;
    q.a=reinterpret_cast<const __half*>(0x1001); require(validate_quad_mma(q)==cudaErrorInvalidValue,"alignment"); q.a=a;
    q.a=reinterpret_cast<const __half*>(UINTPTR_MAX-1); require(validate_quad_mma(q)==cudaErrorInvalidValue,"address overflow");
    q={nullptr,nullptr,nullptr,0,0,0,0}; require(launch_quad_mma_checked(q,nullptr)==cudaSuccess,"empty no CUDA calls");
    q={a,b,y,UINT64_MAX,UINT64_MAX,UINT64_MAX,UINT32_MAX};
    // Large metadata must not wrap into a small capacity requirement.
    q.y_elements=64; require(validate_quad_mma(q)==cudaErrorInvalidValue,"large count capacity");
}
int main(int argc,char** argv) {
    host_checks();
    if(argc==2 && std::strcmp(argv[1],"--host")==0) { std::puts("quad host admission PASS"); return 0; }
    int device=0; cudaDeviceProp prop{}; check(cudaGetDevice(&device)); check(cudaGetDeviceProperties(&prop,device));
    require(prop.major>=7,"SM70 or newer required");
    cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    std::size_t outputs=0; int fixtures=0;
    for(unsigned p : {1u,3u,4u,7u,16u,17u}) {
        std::vector<__half> a(p*32), b(p*32); std::vector<float> y(p*64+4), expected(p*64);
        __half *da,*db; float* dy;
        check(cudaMalloc(&da,a.size()*sizeof(__half))); check(cudaMalloc(&db,b.size()*sizeof(__half)));
        check(cudaMalloc(&dy,y.size()*sizeof(float)));
        QuadMmaRequest q{da,db,dy,a.size(),b.size(),p*64,p};
        auto bad=q; bad.y_elements--; require(launch_quad_mma_checked(bad,stream)==cudaErrorInvalidValue,"launch capacity rejected");
        bad=q; bad.y=reinterpret_cast<float*>(da); require(launch_quad_mma_checked(bad,stream)==cudaErrorInvalidValue,"launch alias rejected");
        for(int fixture=-1;fixture<32;++fixture) {
            for(unsigned panel=0;panel<p;++panel) {
                for(unsigned row=0;row<8;++row) for(unsigned k=0;k<4;++k)
                    a[panel*32+row*4+k]=__float2half(fixture<0 ? float((panel+1)*(row+1)+k)/8.f : (k==unsigned(fixture/8) ? float(panel+row+1) : 0.f));
                for(unsigned k=0;k<4;++k) for(unsigned col=0;col<8;++col)
                    b[panel*32+k*8+col]=__float2half(fixture<0 ? float((panel+2)*(col+1)+k)/16.f : (k==unsigned(fixture/8) && col==unsigned(fixture%8) ? float(panel+1) : 0.f));
                for(unsigned row=0;row<8;++row) for(unsigned col=0;col<8;++col) {
                    float sum=0; for(unsigned k=0;k<4;++k) sum+=__half2float(a[panel*32+row*4+k])*__half2float(b[panel*32+k*8+col]);
                    expected[panel*64+row*8+col]=sum;
                }
            }
            std::fill(y.begin(),y.end(),-12345.f);
            check(cudaMemcpyAsync(da,a.data(),a.size()*sizeof(__half),cudaMemcpyHostToDevice,stream));
            check(cudaMemcpyAsync(db,b.data(),b.size()*sizeof(__half),cudaMemcpyHostToDevice,stream));
            check(cudaMemcpyAsync(dy,y.data(),y.size()*sizeof(float),cudaMemcpyHostToDevice,stream));
            check(launch_quad_mma_checked(q,stream));
            check(cudaMemcpyAsync(y.data(),dy,y.size()*sizeof(float),cudaMemcpyDeviceToHost,stream)); check(cudaStreamSynchronize(stream));
            for(std::size_t i=0;i<expected.size();++i) {
                if(!std::isfinite(y[i]) || std::fabs(y[i]-expected[i])>1e-4f) {
                    std::fprintf(stderr,"count=%u fixture=%d index=%zu actual=%g expected=%g\n",p,fixture,i,y[i],expected[i]); return 1;
                }
            }
            for(std::size_t i=expected.size();i<y.size();++i) require(y[i]==-12345.f,"tail overwrite");
            outputs+=expected.size(); ++fixtures;
        }
        check(cudaFree(da)); check(cudaFree(db)); check(cudaFree(dy));
    }
    check(cudaStreamDestroy(stream));
    std::printf("quad GPU PASS device=%s sm=%d%d fixtures=%d outputs=%zu FP16 inputs FP32 accumulation/output\n",prop.name,prop.major,prop.minor,fixtures,outputs);
}
