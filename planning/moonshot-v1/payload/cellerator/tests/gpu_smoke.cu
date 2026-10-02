#include <moonshot/kernels.cuh>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace cellerator::experimental::moonshot;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> struct Dev{
 T* p=nullptr; explicit Dev(std::size_t n){check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));}
 ~Dev(){if(p)cudaFree(p);} Dev(const Dev&)=delete;
};
int main(){try{
 cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0));
 if(prop.major!=7||prop.minor!=0){std::cerr<<"This smoke is deliberately scoped to sm70\n";return 77;}
 constexpr unsigned n=7; // Exercises partial four-job packet.
 std::vector<__half>a(n*32),b(n*32);std::vector<float>y(n*64);
 for(unsigned i=0;i<a.size();++i){a[i]=__float2half((int(i%11)-5)*0.03125f);b[i]=__float2half((int(i%7)-3)*0.0625f);}
 Dev<__half>da(a.size()),db(b.size());Dev<float>dy(y.size());
 check(cudaMemcpy(da.p,a.data(),a.size()*2,cudaMemcpyHostToDevice));
 check(cudaMemcpy(db.p,b.data(),b.size()*2,cudaMemcpyHostToDevice));
 check(launch_quad_mma(da.p,db.p,dy.p,n,nullptr));check(cudaDeviceSynchronize());
 check(cudaMemcpy(y.data(),dy.p,y.size()*4,cudaMemcpyDeviceToHost));
 for(unsigned p=0;p<n;++p)for(unsigned i=0;i<8;++i)for(unsigned j=0;j<8;++j){
  float want=0;for(unsigned k=0;k<4;++k)want+=__half2float(a[p*32+i*4+k])*__half2float(b[p*32+k*8+j]);
  if(std::abs(y[p*64+i*8+j]-want)>1e-5f)throw std::runtime_error("quad mapping mismatch");
 }
 std::cout<<"quad_mma: passed on sm70; other kernels not exercised by this smoke\n";
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
