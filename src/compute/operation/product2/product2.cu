#include <Cellerator/compute/operation/product2/c_api.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <limits>
#include <new>
#include <memory>
#include <vector>

struct ce_product2_cuda_context {
 uint64_t n{},m{},generation{};
 int64_t *a{},*b{};
 std::vector<int64_t> host_a,host_b;
 int device{-1}; cudaStream_t stream{};
 ~ce_product2_cuda_context() {
  if(a||b) { int old=-1; cudaGetDevice(&old); cudaSetDevice(device);
   cudaStreamSynchronize(stream); cudaFree(a);cudaFree(b);
   if(old>=0 && old!=device) cudaSetDevice(old);
  }
 }
};
namespace {
ce_product2_status topology_extent(const int64_t*p,uint64_t count,uint64_t used) {
 if(count>SIZE_MAX/sizeof(int64_t))return CE_PRODUCT2_OVERFLOW;
 auto address=reinterpret_cast<uintptr_t>(p);
 if(used && (!p || address%alignof(int64_t)))return CE_PRODUCT2_INVALID_ARGUMENT;
 if(used>(UINTPTR_MAX-address)/sizeof(int64_t))return CE_PRODUCT2_OVERFLOW;
 return count<used?CE_PRODUCT2_INVALID_ARGUMENT:CE_PRODUCT2_SUCCESS;
}
struct Span { uintptr_t first,last; };
bool span(const void *p,uint64_t count,Span& s) {
 s.first=reinterpret_cast<uintptr_t>(p);
 if(count && (!p || s.first%alignof(float))) return false;
 if(count>(UINTPTR_MAX-s.first)/sizeof(float)) return false;
 s.last=s.first+count*sizeof(float);return true;
}
bool overlap(Span a,Span b) {return a.first<a.last && b.first<b.last && a.first<b.last && b.first<a.last;}
ce_product2_status device_span(const void*p,uint64_t count,int dev) {
 if(!count)return CE_PRODUCT2_SUCCESS;
 cudaPointerAttributes attr{};
 if(cudaPointerGetAttributes(&attr,p)!=cudaSuccess)return CE_PRODUCT2_INVALID_ARGUMENT;
 if(attr.type!=cudaMemoryTypeDevice || attr.device!=dev)return CE_PRODUCT2_INVALID_ARGUMENT;
 CUdeviceptr base{};size_t bytes{};auto ptr=reinterpret_cast<CUdeviceptr>(p);
 if(cuMemGetAddressRange(&base,&bytes,ptr)!=CUDA_SUCCESS || ptr<base || ptr-base>bytes || count>(bytes-(ptr-base))/sizeof(float))return CE_PRODUCT2_INVALID_ARGUMENT;
 return CE_PRODUCT2_SUCCESS;
}
ce_product2_status admit(const ce_product2_cuda_context*c,const ce_product2_binding*v,void*s,int mode) {
 if(!c||!v || reinterpret_cast<cudaStream_t>(s)!=c->stream)return CE_PRODUCT2_INVALID_ARGUMENT;
 if(v->expected_structure_generation!=c->generation || v->current_structure_generation!=c->generation || v->expected_value_generation!=v->current_value_generation || v->expected_parameter_generation!=v->current_parameter_generation)return CE_PRODUCT2_STALE_GENERATION;
 if(v->x_count<c->n || v->k_count<c->m)return CE_PRODUCT2_INVALID_ARGUMENT;
 const void* rp[5]={v->x,v->k,nullptr,nullptr,nullptr};uint64_t rc[5]={v->x_count,v->k_count,0,0,0};
 void*wp[2]={};uint64_t wc[2]={};int reads=2,writes=1;
 if(mode==0){wp[0]=v->y;wc[0]=v->y_count;if(wc[0]<c->m)return CE_PRODUCT2_INVALID_ARGUMENT;}
 if(mode==1){rp[2]=v->g;rc[2]=v->g_count;reads=3;wp[0]=v->gx;wc[0]=v->gx_count;wp[1]=v->gk;wc[1]=v->gk_count;writes=2;if(rc[2]<c->m||wc[0]<c->n||wc[1]<c->m)return CE_PRODUCT2_INVALID_ARGUMENT;}
 if(mode==2){rp[2]=v->dx;rc[2]=v->dx_count;rp[3]=v->dk;rc[3]=v->dk_count;reads=4;wp[0]=v->dy;wc[0]=v->dy_count;if(rc[2]<c->n||rc[3]<c->m||wc[0]<c->m)return CE_PRODUCT2_INVALID_ARGUMENT;}
 Span rs[5],ws[2];for(int i=0;i<reads;++i)if(!span(rp[i],rc[i],rs[i]))return CE_PRODUCT2_OVERFLOW;
 for(int i=0;i<writes;++i){if(!span(wp[i],wc[i],ws[i]))return CE_PRODUCT2_OVERFLOW;for(int j=0;j<reads;++j)if(overlap(ws[i],rs[j]))return CE_PRODUCT2_ALIAS;for(int j=0;j<i;++j)if(overlap(ws[i],ws[j]))return CE_PRODUCT2_ALIAS;}
 int dev=-1,sd=-1;if(cudaGetDevice(&dev)!=cudaSuccess || dev!=c->device)return CE_PRODUCT2_BACKEND_ERROR;
 if(cudaStreamGetDevice(c->stream,&sd)!=cudaSuccess || sd!=dev)return CE_PRODUCT2_BACKEND_ERROR;
 for(int i=0;i<reads;++i){auto e=device_span(rp[i],rc[i],dev);if(e)return e;}
 for(int i=0;i<writes;++i){auto e=device_span(wp[i],wc[i],dev);if(e)return e;}
 return CE_PRODUCT2_SUCCESS;
}
__device__ float gather(const float*x,int64_t id) {
 auto peers=__match_any_sync(0xffffffffu,static_cast<unsigned long long>(id));
 int leader=__ffs(peers)-1;float value=0.f;
 if((threadIdx.x&31)==leader && id>=0)value=x[id];
 return __shfl_sync(0xffffffffu,value,leader);
}
__global__ void calculate(uint64_t m,const int64_t*a,const int64_t*b,ce_product2_binding v,int mode) {
 uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;bool live=i<m;
 int64_t ai=live?a[i]:-1,bi=live?b[i]:-1;
 float xa=gather(v.x,ai),xb=gather(v.x,bi);
 if(mode==2){float da=gather(v.dx,ai),db=gather(v.dx,bi);if(live)v.dy[i]=v.dk[i]*(xa*xb)+v.k[i]*fmaf(da,xb,xa*db);}
 else if(live && mode==0)v.y[i]=(v.k[i]*xa)*xb;
 else if(live){float g=v.g[i],k=v.k[i];atomicAdd(v.gx+ai,(g*k)*xb);atomicAdd(v.gx+bi,(g*k)*xa);v.gk[i]=(g*xa)*xb;}
}
ce_product2_status run(const ce_product2_cuda_context*c,const ce_product2_binding*v,void*s,int mode) {
 auto e=admit(c,v,s,mode);if(e)return e;
 if(mode==1 && c->n && cudaMemsetAsync(v->gx,0,c->n*sizeof(float),c->stream)!=cudaSuccess)return CE_PRODUCT2_BACKEND_ERROR;
 if(!c->m)return CE_PRODUCT2_SUCCESS;
 calculate<<<static_cast<unsigned>((c->m+127)/128),128,0,c->stream>>>(c->m,c->a,c->b,*v,mode);
 return cudaGetLastError()==cudaSuccess?CE_PRODUCT2_SUCCESS:CE_PRODUCT2_BACKEND_ERROR;
}
}
extern "C" ce_product2_status ce_product2_cuda_create(uint64_t n,uint64_t m,const int64_t*a,uint64_t ac,const int64_t*b,uint64_t bc,uint64_t generation,void*s,ce_product2_cuda_context**out) {
 if(!out)return CE_PRODUCT2_INVALID_ARGUMENT;
 auto ae=topology_extent(a,ac,m);if(ae)return ae;
 auto be=topology_extent(b,bc,m);if(be)return be;
 if(m && !n)return CE_PRODUCT2_INVALID_ARGUMENT;
 if(n>uint64_t(INT64_MAX) || m>uint64_t(2147483647)*128 || m>SIZE_MAX/sizeof(int64_t)||n>SIZE_MAX/sizeof(float))return CE_PRODUCT2_OVERFLOW;
 for(uint64_t i=0;i<m;++i)if(a[i]<0||b[i]<0||uint64_t(a[i])>=n||uint64_t(b[i])>=n)return CE_PRODUCT2_INVALID_INDEX;
 try {
 auto c=std::make_unique<ce_product2_cuda_context>();if(m){c->host_a.assign(a,a+m);c->host_b.assign(b,b+m);}c->n=n;c->m=m;c->generation=generation;c->stream=reinterpret_cast<cudaStream_t>(s);
 int sd=-1;cudaDeviceProp p{};
 if(cudaGetDevice(&c->device)!=cudaSuccess || cudaStreamGetDevice(c->stream,&sd)!=cudaSuccess || sd!=c->device || cudaGetDeviceProperties(&p,c->device)!=cudaSuccess)return CE_PRODUCT2_BACKEND_ERROR;
 if(p.major!=7 || p.minor!=0)return CE_PRODUCT2_UNAVAILABLE;
 if(m){if(cudaMalloc(&c->a,m*sizeof(int64_t))!=cudaSuccess || cudaMalloc(&c->b,m*sizeof(int64_t))!=cudaSuccess)return CE_PRODUCT2_BACKEND_ERROR;
 auto ea=cudaMemcpyAsync(c->a,a,m*sizeof(int64_t),cudaMemcpyHostToDevice,c->stream);
 auto eb=ea==cudaSuccess?cudaMemcpyAsync(c->b,b,m*sizeof(int64_t),cudaMemcpyHostToDevice,c->stream):ea;
 auto es=cudaStreamSynchronize(c->stream);if(ea!=cudaSuccess||eb!=cudaSuccess||es!=cudaSuccess)return CE_PRODUCT2_BACKEND_ERROR;}
 *out=c.release();return CE_PRODUCT2_SUCCESS;
 }catch(const std::bad_alloc&){return CE_PRODUCT2_BACKEND_ERROR;}
}
extern "C" void ce_product2_cuda_destroy(ce_product2_cuda_context*c){delete c;}
extern "C" ce_product2_status ce_product2_cuda_forward(const ce_product2_cuda_context*c,const ce_product2_binding*v,void*s){return run(c,v,s,0);}
extern "C" ce_product2_status ce_product2_cuda_vjp(const ce_product2_cuda_context*c,const ce_product2_binding*v,void*s){return run(c,v,s,1);}
extern "C" ce_product2_status ce_product2_cuda_jvp(const ce_product2_cuda_context*c,const ce_product2_binding*v,void*s){return run(c,v,s,2);}
extern "C" ce_product2_status ce_product2_cuda_admit(const ce_product2_cuda_context*c,const ce_product2_binding*v,void*s,int operation){if(operation<0||operation>2)return CE_PRODUCT2_INVALID_ARGUMENT;return admit(c,v,s,operation);}

extern "C" ce_product2_status ce_product2_cuda_matches(const ce_product2_cuda_context*c,uint64_t n,uint64_t m,const int64_t*a,uint64_t ac,const int64_t*b,uint64_t bc,uint64_t gen) {
 if(!c)return CE_PRODUCT2_INVALID_ARGUMENT;
 if(c->n!=n || c->m!=m)return CE_PRODUCT2_INVALID_ARGUMENT;
 if(c->generation!=gen)return CE_PRODUCT2_STALE_GENERATION;
 auto ae=topology_extent(a,ac,m);if(ae)return ae;
 auto be=topology_extent(b,bc,m);if(be)return be;
 for(uint64_t i=0;i<m;++i)if(c->host_a[i]!=a[i] || c->host_b[i]!=b[i])return CE_PRODUCT2_INVALID_ARGUMENT;
 return CE_PRODUCT2_SUCCESS;
}
