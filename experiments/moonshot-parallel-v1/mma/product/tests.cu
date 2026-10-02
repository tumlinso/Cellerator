#include "product.hpp"
#include <cuda_runtime.h>
#include <cuda.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
using namespace cellerator::experimental::moonshot::product;
void require(bool ok,const char* msg) { if(!ok) { std::fprintf(stderr,"FAIL: %s\n",msg); std::exit(1); } }
void cuda_ok(cudaError_t e) { if(e!=cudaSuccess) { std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));std::exit(1); } }
bool close(float a,float b) { return std::abs(a-b)<=2e-6f*(1.f+std::max(std::abs(a),std::abs(b))); }
template<class T> struct Device {
    T* p{};std::size_t n;
    explicit Device(std::size_t size):n(size) { cuda_ok(cudaMalloc(&p,size*sizeof(T))); }
    ~Device() { cudaFree(p); }
    void copy(const std::vector<T>& a) { cuda_ok(cudaMemcpy(p,a.data(),n*sizeof(T),cudaMemcpyHostToDevice)); }
    std::vector<T> read() { std::vector<T>a(n);cuda_ok(cudaMemcpy(a.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return a; }
};
void host_tests() {
    float x[]={0,3,-2},v[]={2,-1,4},k[2]={5,2},y[3],dy[2];std::uint32_t a[2]={0,1},b[2]={1,1};
    Inputs in{{x,3},{v,3},{k,2},{a,2},{b,2},2};Outputs out{{y,2},{dy,2}};
    require(validate_metadata(in,out)==cudaSuccess,"valid host metadata");
    oracle(x[0],x[1],v[0],v[1],k[0],y[0],dy[0]);
    require(y[0]==0 && dy[0]==30,"zero-safe one zero derivative");
    oracle(0,0,2,3,5,y[0],dy[0]);require(y[0]==0 && dy[0]==0,"two zero derivative");
    oracle(3,3,-1,-1,2,y[0],dy[0]);require(y[0]==18 && dy[0]==-12,"repeated argument multiplicity");
    oracle(-2,3,4,-1,0,y[0],dy[0]);require(y[0]==0 && dy[0]==0,"zero coefficient");
    auto bad=out;bad.jvp=bad.value;require(validate_metadata(in,bad)==cudaErrorInvalidValue,"equal outputs rejected");
    bad=out;bad.value={x+1,2};require(validate_metadata(in,bad)==cudaErrorInvalidValue,"input overlap rejected");
    bad=out;bad.jvp={y+1,2};require(validate_metadata(in,bad)==cudaErrorInvalidValue,"partial output overlap rejected");
    bad=out;bad.value.size=1;require(validate_metadata(in,bad)==cudaErrorInvalidValue,"short output rejected");
    auto badin=in;badin.direction.size=2;require(validate_metadata(badin,out)==cudaErrorInvalidValue,"mismatched state extent rejected");
    badin=in;badin.src1.size=1;require(validate_metadata(badin,out)==cudaErrorInvalidValue,"short index extent rejected");
    badin=in;badin.x.data=reinterpret_cast<float*>(reinterpret_cast<std::uintptr_t>(x)+1);
    require(validate_metadata(badin,out)==cudaErrorInvalidValue,"unaligned pointer rejected");
    badin=in;badin.x.size=SIZE_MAX;badin.direction.size=SIZE_MAX;
    require(validate_metadata(badin,out)==cudaErrorInvalidValue,"byte range overflow rejected");
    Prepared p;Generations gen{1,2,3,4};
    require(launch_product2(p,gen,nullptr)==cudaErrorInvalidValue,"unprepared rejected");
    cuda_ok(prepare_product2({}, {},gen,nullptr,p));cuda_ok(launch_product2(p,gen,nullptr));
    gen.value++;require(launch_product2(p,gen,nullptr)==cudaErrorInvalidValue,"stale generation rejected");
    std::puts("product host checks PASS");
}
void gpu_tests() {
    cudaStream_t stream;cuda_ok(cudaStreamCreate(&stream));
    Generations gen{10,20,30,40};
    const std::vector<float>x={0,3,-2,0.5f,1,0,-0.75f},v={2,-1,4,-3,0.5f,7,1.25f};
    Device<float> dx(x.size()),dv(v.size());dx.copy(x);dv.copy(v);
    for(unsigned n:{0u,1u,31u,32u,33u,127u,129u}) {
        // Extra output element checks that the last warp never writes padding.
        const auto capacity=std::size_t(n)+1;
        Device<float> dk(capacity),dy(capacity),dj(capacity);
        Device<std::uint32_t>da(capacity),db(capacity);
        std::vector<float>k(capacity),sentinel(capacity,12345.f);
        std::vector<std::uint32_t>a(capacity),b(capacity);
        for(unsigned i=0;i<n;++i) { a[i]=i%x.size();b[i]=(i%3==0)?a[i]:(i+1)%x.size();k[i]=(i%4==0)?0.f:(float(int(i%5)-2)*0.3f); }
        if(n) { a[0]=0;b[0]=1;k[0]=5; }
        dk.copy(k);dy.copy(sentinel);dj.copy(sentinel);da.copy(a);db.copy(b);
        Inputs in{{dx.p,x.size()},{dv.p,v.size()},{dk.p,capacity},{da.p,capacity},{db.p,capacity},n};
        Outputs out{{dy.p,capacity},{dj.p,capacity}};Prepared p;
        cuda_ok(prepare_product2(in,out,gen,stream,p));cuda_ok(launch_product2(p,gen,stream));cuda_ok(cudaStreamSynchronize(stream));
        auto y=dy.read(),j=dj.read();
        for(unsigned i=0;i<n;++i) {
            float ey,ej;oracle(x[a[i]],x[b[i]],v[a[i]],v[b[i]],k[i],ey,ej);
            require(close(y[i],ey) && close(j[i],ej),"GPU vs FP32 oracle");
            const double eps=1e-4;
            const double plus=k[i]*(double(x[a[i]])+eps*v[a[i]])*(double(x[b[i]])+eps*v[b[i]]);
            const double minus=k[i]*(double(x[a[i]])-eps*v[a[i]])*(double(x[b[i]])-eps*v[b[i]]);
            require(std::abs(double(j[i])-(plus-minus)/(2*eps))<2e-5*(1+std::abs(j[i])),"independent finite difference JVP");
        }
        require(y[n]==12345.f && j[n]==12345.f,"padding guard");
        if(n) {
            CUdeviceptr allocation_base{};std::size_t allocation_bytes{};
            const auto output_ptr=reinterpret_cast<CUdeviceptr>(dy.p);
            require(cuMemGetAddressRange(&allocation_base,&allocation_bytes,output_ptr)==CUDA_SUCCESS,"query actual allocation extent");
            auto oversized=out;
            oversized.value.size=(allocation_bytes-(output_ptr-allocation_base))/sizeof(float)+1;
            // A declared range beyond the actual allocation can also intersect
            // another input/output span, which metadata rejects first.
            require(prepare_product2(in,oversized,gen,stream,p)!=cudaSuccess,"allocation extent rejected");
            auto aliases=out;aliases.jvp=aliases.value;
            require(prepare_product2(in,aliases,gen,stream,p)==cudaErrorInvalidValue,"device output alias rejected");
            require(launch_product2(p,gen,stream)==cudaErrorInvalidValue,"failed prepare clears snapshot");
            aliases=out;aliases.value={dx.p,x.size()};
            require(prepare_product2(in,aliases,gen,stream,p)==cudaErrorInvalidValue,"device input alias rejected");
            a[0]=padding_id;da.copy(a);
            require(prepare_product2(in,out,gen,stream,p)==cudaErrorInvalidValue,"live sentinel rejected");
            a[0]=x.size();da.copy(a);
            require(prepare_product2(in,out,gen,stream,p)==cudaErrorInvalidValue,"index bounds rejected");
            a[0]=0;da.copy(a);
            auto host=in;host.x={x.data(),x.size()};
            require(prepare_product2(host,out,gen,stream,p)!=cudaSuccess,"host memory rejected");
            cuda_ok(prepare_product2(in,out,gen,stream,p));
            auto stale=gen;stale.structure++;
            require(launch_product2(p,stale,stream)==cudaErrorInvalidValue,"stale structure rejected");
        }
        std::printf("product GPU count=%u PASS\n",n);
    }
    cuda_ok(cudaStreamDestroy(stream));
}
int main(int argc,char**argv) { host_tests();if(argc>1 && std::string(argv[1])=="--gpu")gpu_tests();return 0; }
