#include "patch16.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace cellerator::experimental::moonshot;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void host_checks() {
    patch16::Request q;
    require(patch16::validate(q) == cudaSuccess, "empty request");
    q.compute_major = 6;
    require(patch16::validate(q) == cudaErrorNotSupported, "architecture rejection");
    q.compute_major = 7; q.patches = 1;
    require(patch16::validate(q) == cudaErrorInvalidValue, "capacity rejection");
    q.l_capacity = q.x_capacity = q.r_capacity = q.y_capacity = 256;
    require(patch16::validate(q) == cudaErrorInvalidValue, "null rejection");
    // Addresses are never dereferenced by the pure metadata validator.
    q.l = reinterpret_cast<const __half*>(0x10000);
    q.x = reinterpret_cast<const __half*>(0x20000);
    q.r = reinterpret_cast<const __half*>(0x30000);
    q.y = reinterpret_cast<float*>(0x40000);
    require(patch16::validate(q) == cudaSuccess, "valid metadata");
    q.y = reinterpret_cast<float*>(0x10020);
    require(patch16::validate(q) == cudaErrorInvalidValue, "output alias rejection");
    q.y = reinterpret_cast<float*>(0x40004);
    require(patch16::validate(q) == cudaErrorInvalidValue, "alignment rejection");
    q.y = reinterpret_cast<float*>(0x40000); q.device = -1;
    require(patch16::validate(q) == cudaErrorInvalidDevice, "device rejection");
    std::cout << "patch16 host admission passed\n";
}
template<class T> struct Device {
    T* p{};
    explicit Device(std::size_t n) { check(cudaMalloc(reinterpret_cast<void**>(&p), n*sizeof(T))); }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
};
void gpu_case(unsigned count, int kind) {
    const std::size_t n = std::size_t(count)*256;
    std::vector<__half> l(n), x(n), r(n);
    std::vector<float> actual(n), expected(n);
    for (unsigned p=0;p<count;++p) for (unsigned i=0;i<16;++i) for (unsigned j=0;j<16;++j) {
        auto at = p*256+i*16+j;
        l[at] = __float2half_rn(kind == 1 ? float(i == j) : (int((i*3+j*5+p)%13)-6)*0.125f);
        x[at] = __float2half_rn((int((i*7+j*2+p)%11)-5)*(kind == 2 ? 8.f : 0.0625f));
        r[at] = __float2half_rn(kind == 1 ? float(i == j) : (int((i*2+j*7+p)%9)-4)*0.125f);
    }
    for (unsigned p=0;p<count;++p) {
        float hidden[256];
        for (unsigned i=0;i<16;++i) for (unsigned j=0;j<16;++j) {
            float t=0;
            for (unsigned k=0;k<16;++k)
                t += __half2float(l[p*256+i*16+k])*__half2float(x[p*256+k*16+j]);
            hidden[i*16+j] = __half2float(__float2half_rn(std::tanh(t)));
        }
        for (unsigned i=0;i<16;++i) for (unsigned j=0;j<16;++j) {
            float y=0;
            for (unsigned k=0;k<16;++k) y += hidden[i*16+k]*__half2float(r[p*256+k*16+j]);
            expected[p*256+i*16+j]=y;
        }
    }
    Device<__half> dl(n), dx(n), dr(n); Device<float> dy(n+256);
    check(cudaMemcpy(dl.p,l.data(),n*2,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dx.p,x.data(),n*2,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dr.p,r.data(),n*2,cudaMemcpyHostToDevice));
    // The extra patch must remain untouched by partial blocks.
    std::vector<float> init(n+256,-1234.f);
    check(cudaMemcpy(dy.p,init.data(),init.size()*4,cudaMemcpyHostToDevice));
    patch16::Request request{dl.p,dx.p,dr.p,dy.p,n,n,n,n,count,0,7,0};
    cudaStream_t stream{}; check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(patch16::launch_prepared(request,stream));
    check(cudaStreamSynchronize(stream)); check(cudaStreamDestroy(stream));
    check(cudaMemcpy(init.data(),dy.p,init.size()*4,cudaMemcpyDeviceToHost));
    for (std::size_t i=0;i<n;++i)
        require(std::isfinite(init[i]) && std::abs(init[i]-expected[i]) <= 0.003f + 0.002f*std::abs(expected[i]),
                "same stored half oracle mismatch or nonfinite output");
    for (std::size_t i=n;i<init.size();++i) require(init[i] == -1234.f,"partial block overwrite");
    std::cout << "patch16 oracle passed count=" << count << " kind=" << kind << '\n';
}
int main(int argc,char** argv) { try {
    host_checks();
    if (argc>1 && std::string(argv[1]) == "--host-only") return 0;
    int devices{}; auto e=cudaGetDeviceCount(&devices);
    if (e==cudaErrorNoDevice || e==cudaErrorInsufficientDriver || !devices) return 77;
    check(e); check(cudaSetDevice(0));
    cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop,0));
    if (prop.major!=7 || prop.minor!=0) return 77;
    check(launch_patch16(nullptr,nullptr,nullptr,nullptr,0,nullptr));
    for (unsigned n : {1u,4u,5u}) for (int kind : {0,1,2}) gpu_case(n,kind);
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; } }
