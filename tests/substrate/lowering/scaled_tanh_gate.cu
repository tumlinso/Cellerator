#include <Cellerator/compiler/substrate/guarded_scaled_tanh.hh>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>
using namespace cellerator::compiler::substrate;
static void check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e)); std::exit(1); }
}
static void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr,"FAIL: %s\n",message); std::exit(2); }
}
static bool same(float a, float b) { return a == b || (std::isnan(a) && std::isnan(b)); }
int main() {
    int device; check(cudaGetDevice(&device));
    cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop,device));
    require(prop.major == 7 && prop.minor == 0,"qualification requires SM70");
    constexpr std::size_t n = 262147; // Also exercises a partial final block.
    std::vector<float> a(n),p(n),z(n),y(n),reference_z(n),reference_y(n);
    for (std::size_t i=0;i<n;++i) { a[i] = (int(i%127)-63)*0.03125f; p[i] = (int(i%31)-15)*0.0625f; }
    a[0]=0; p[0]=std::numeric_limits<float>::infinity();
    a[1]=-0.0f; p[1]=1; a[2]=std::numeric_limits<float>::infinity(); p[2]=1;
    a[3]=std::numeric_limits<float>::quiet_NaN();
    float *da,*dp,*dz,*dy; const auto bytes=n*sizeof(float);
    check(cudaMalloc(&da,bytes)); check(cudaMalloc(&dp,bytes));
    check(cudaMalloc(&dz,bytes)); check(cudaMalloc(&dy,bytes));
    check(cudaMemcpy(da,a.data(),bytes,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dp,p.data(),bytes,cudaMemcpyHostToDevice));
    device_scaled_tanh_binding b{da,dp,dz,dy,n};
    scaled_tanh_plan plan{{n,17}}; realization selected{};
    check(launch_scaled_tanh(realization::direct,b,nullptr));
    check(cudaMemcpy(reference_z.data(),dz,bytes,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(reference_y.data(),dy,bytes,cudaMemcpyDeviceToHost));
    check(launch_guarded_scaled_tanh(plan,{n,17},b,nullptr,&selected));
    require(selected==realization::fused_scaled_tanh,"matching guard selects fusion");
    check(cudaMemcpy(z.data(),dz,bytes,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(y.data(),dy,bytes,cudaMemcpyDeviceToHost));
    for (std::size_t i=0;i<n;++i) {
        require(same(z[i],reference_z[i]) && same(y[i],reference_y[i]),"direct/fused equality");
        float expected_z=a[i]*p[i], expected_y=std::tanh(expected_z);
        require(same(z[i],expected_z),"host multiply reference");
        require(same(y[i],expected_y) || std::abs(y[i]-expected_y)<=2e-7f,"host tanh tolerance");
        if (i==1) require(std::signbit(z[i]) && std::signbit(y[i]),"signed zero");
    }
    check(launch_guarded_scaled_tanh(plan,{n,18},b,nullptr,&selected));
    require(selected==realization::direct,"support generation fallback");
    auto smaller=b; --smaller.extent;
    check(launch_guarded_scaled_tanh(plan,{n-1,17},smaller,nullptr,&selected));
    require(selected==realization::direct,"shape fallback");
    require(launch_guarded_scaled_tanh(plan,{n,17,numerical_policy::approximate},b,nullptr)==cudaErrorInvalidValue,
        "changed arithmetic requires reprepare");
    auto alias=b; alias.output=dz;
    require(launch_scaled_tanh(realization::fused_scaled_tanh,alias,nullptr)==cudaErrorInvalidValue,"reject alias");
    // Value changes do not invalidate structural specialization, and must be read live.
    p[5]=2.0f; check(cudaMemcpy(dp+5,p.data()+5,sizeof(float),cudaMemcpyHostToDevice));
    check(launch_guarded_scaled_tanh(plan,{n,17},b,nullptr,&selected));
    check(cudaMemcpy(y.data()+5,dy+5,sizeof(float),cudaMemcpyDeviceToHost));
    require(std::abs(y[5]-std::tanh(a[5]*p[5]))<=2e-7f,"live trainable parameter");
    cudaEvent_t start,stop; check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
    auto timing=[&](realization r) {
        for(int i=0;i<5;++i) check(launch_scaled_tanh(r,b,nullptr));
        check(cudaEventRecord(start));
        for(int i=0;i<100;++i) check(launch_scaled_tanh(r,b,nullptr));
        check(cudaEventRecord(stop)); check(cudaEventSynchronize(stop));
        float ms; check(cudaEventElapsedTime(&ms,start,stop)); return ms/100;
    };
    float direct=timing(realization::direct), fused=timing(realization::fused_scaled_tanh);
    std::printf("SM70 scaled_tanh correctness PASS elements=%zu direct_ms=%.6f fused_ms=%.6f ratio=%.4f\n",n,direct,fused,fused/direct);
    check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
    check(cudaFree(da)); check(cudaFree(dp)); check(cudaFree(dz)); check(cudaFree(dy));
}
