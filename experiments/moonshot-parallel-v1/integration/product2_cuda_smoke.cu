#include <Cellerator/compute/operation/product2/product2.hh>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <limits>
void require(bool b,const char*m){if(!b)throw std::runtime_error(m);}
void ok(cudaError_t e){require(e==cudaSuccess,cudaGetErrorString(e));}
void success(ce_product2_status e){require(e==CE_PRODUCT2_SUCCESS,"native CUDA admission/launch failed");}
struct Buffer {float*p{};size_t n{};explicit Buffer(size_t size):n(size){if(n)ok(cudaMalloc(&p,n*sizeof(float)));}~Buffer(){cudaFree(p);}void set(const std::vector<float>&v){if(n)ok(cudaMemcpy(p,v.data(),n*sizeof(float),cudaMemcpyHostToDevice));}std::vector<float> get(){std::vector<float>v(n);if(n)ok(cudaMemcpy(v.data(),p,n*sizeof(float),cudaMemcpyDeviceToHost));return v;}};
void near(const std::vector<float>&a,const std::vector<float>&b){require(a.size()==b.size(),"shape");for(size_t i=0;i<a.size();++i)require(std::abs(a[i]-b[i])<=2e-5f*(1+std::abs(b[i])),"numerical mismatch");}
void check(size_t m,cudaStream_t stream){
 const size_t n=3;std::vector<int64_t>a(m),b(m);std::vector<float>x={0.f,2.f,-3.f},dx={.5f,-.25f,1.f},k(m),dk(m),g(m),y(m),dy(m),gx(n),gk(m);
 for(size_t i=0;i<m;++i){a[i]=i%3;b[i]=i%2?int64_t(i%3):int64_t((i+1)%3);k[i]=float(int(i%7)-3)*.25f;dk[i]=.125f;g[i]=float(int(i%5)-2)*.5f;
 y[i]=k[i]*x[a[i]]*x[b[i]];dy[i]=dk[i]*x[a[i]]*x[b[i]]+k[i]*(dx[a[i]]*x[b[i]]+x[a[i]]*dx[b[i]]);
 gx[a[i]]+=g[i]*k[i]*x[b[i]];gx[b[i]]+=g[i]*k[i]*x[a[i]];gk[i]=g[i]*x[a[i]]*x[b[i]];}
 Buffer X(n),K(m),Y(m),G(m),DX(n),DK(m),DY(m),GX(n),GK(m);X.set(x);K.set(k);G.set(g);DX.set(dx);DK.set(dk);GX.set({99,99,99});
 ce_product2_cuda_context*c{};success(ce_product2_cuda_create(n,m,a.data(),m,b.data(),m,7,stream,&c));
 int64_t invalid=-1;auto retained=c;require(ce_product2_cuda_create(n,1,&invalid,1,&invalid,1,7,stream,&c)==CE_PRODUCT2_INVALID_INDEX && c==retained,"failed create lost existing context");
 ce_product2_binding v{};v.x=X.p;v.x_count=n;v.k=K.p;v.k_count=m;v.y=Y.p;v.y_count=m;v.g=G.p;v.g_count=m;v.dx=DX.p;v.dx_count=n;v.dk=DK.p;v.dk_count=m;v.dy=DY.p;v.dy_count=m;v.gx=GX.p;v.gx_count=n;v.gk=GK.p;v.gk_count=m;
 v.expected_structure_generation=v.current_structure_generation=7;v.expected_value_generation=v.current_value_generation=3;v.expected_parameter_generation=v.current_parameter_generation=5;
 success(ce_product2_cuda_forward(c,&v,stream));success(ce_product2_cuda_jvp(c,&v,stream));success(ce_product2_cuda_vjp(c,&v,stream));ok(cudaStreamSynchronize(stream));near(Y.get(),y);near(DY.get(),dy);near(GX.get(),gx);near(GK.get(),gk);near(X.get(),x);near(K.get(),k);
 // Repeated VJP resets the accumulation buffer, including zero live support.
 success(ce_product2_cuda_vjp(c,&v,stream));ok(cudaStreamSynchronize(stream));near(GX.get(),gx);
 auto stale=v;stale.current_value_generation++;
 require(ce_product2_cuda_forward(c,&stale,stream)==CE_PRODUCT2_STALE_GENERATION,"stale value accepted");stale=v;stale.current_parameter_generation++;
 require(ce_product2_cuda_vjp(c,&stale,stream)==CE_PRODUCT2_STALE_GENERATION,"stale parameter accepted");stale=v;stale.current_structure_generation++;
 require(ce_product2_cuda_jvp(c,&stale,stream)==CE_PRODUCT2_STALE_GENERATION,"stale structure accepted");
 if(m){auto alias=v;alias.y=X.p;alias.y_count=m;require(ce_product2_cuda_forward(c,&alias,stream)==CE_PRODUCT2_ALIAS,"input alias accepted");
 alias=v;alias.gk=GX.p;alias.gk_count=m;require(ce_product2_cuda_vjp(c,&alias,stream)==CE_PRODUCT2_ALIAS,"output alias accepted");
 auto host=v;host.x=x.data();require(ce_product2_cuda_forward(c,&host,stream)!=CE_PRODUCT2_SUCCESS,"host pointer accepted");
 CUdeviceptr base{};size_t bytes{};require(cuMemGetAddressRange(&base,&bytes,reinterpret_cast<CUdeviceptr>(X.p))==CUDA_SUCCESS,"extent query");auto oversized=v;oversized.x_count=bytes/sizeof(float)+1;
 require(ce_product2_cuda_forward(c,&oversized,stream)!=CE_PRODUCT2_SUCCESS,"actual allocation extent ignored");
 near(Y.get(),y);near(GX.get(),gx);
 }
 namespace p2=cellerator::compute::product2;namespace ep=cellerator::execution::program;
 p2::prepared_owner owner({n,m,7,a.data(),b.data(),m,m});
 p2::bound_view views[3];for(auto&view:views)view.numeric=v;
 p2::cuda_stage_state states[3]={{&owner,c,p2::operation::forward},{&owner,c,p2::operation::jvp},{&owner,c,p2::operation::vjp}};
 ep::prepared_stage_v2 stages[3];ep::launch_binding_v2 bindings[3];
 for(unsigned i=0;i<3;++i){stages[i]=p2::make_cuda_stage(states[i],i+1,i);bindings[i].input=&views[i];}
 ep::prepared_program_v2 program{};program.stages=stages;program.stage_count=3;
 std::vector<float> sentinel(m,99.f);Y.set(sentinel);DY.set(sentinel);GX.set({99,99,99});
 views[2].numeric.current_parameter_generation++;
 require(ep::execute_prepared_program_v2(program,bindings,3,stream)==ep::program_status::launch_failed,"invalid later stage accepted");
 ok(cudaStreamSynchronize(stream));near(Y.get(),sentinel);near(DY.get(),sentinel);near(GX.get(),{99,99,99});
 views[2].numeric=v;
 if(m){
  auto different=a;different[0]=(different[0]+1)%n;
  p2::prepared_owner mismatched({n,m,7,different.data(),b.data(),m,m});
  states[2].owner=&mismatched;
  require(ep::execute_prepared_program_v2(program,bindings,3,stream)==ep::program_status::launch_failed,"same-shape different topology admitted");
  ok(cudaStreamSynchronize(stream));near(Y.get(),sentinel);near(DY.get(),sentinel);near(GX.get(),{99,99,99});
  states[2].owner=&owner;
 }
 require(ep::execute_prepared_program_v2(program,bindings,3,stream)==ep::program_status::success,"prepared CUDA stage failed");
 ok(cudaStreamSynchronize(stream));near(Y.get(),y);near(DY.get(),dy);near(GX.get(),gx);near(GK.get(),gk);
 // Both native owners copied immutable support before host topology changes.
 for(auto&index:a)index=-1;for(auto&index:b)index=-1;
 require(ep::execute_prepared_program_v2(program,bindings,3,stream)==ep::program_status::success,"copied topology not retained");
 ok(cudaStreamSynchronize(stream));near(Y.get(),y);near(GX.get(),gx);
 ce_product2_cuda_destroy(c);
}
int main(){try{cudaStream_t s{};ok(cudaStreamCreate(&s));for(size_t m:{size_t(0),size_t(1),size_t(5),size_t(129)})check(m,s);
 int64_t bad=-1,good=0;ce_product2_cuda_context*c{};
 require(ce_product2_cuda_create(3,1,&bad,1,&good,1,1,s,&c)==CE_PRODUCT2_INVALID_INDEX,"negative topology accepted");bad=3;
 require(ce_product2_cuda_create(3,1,&bad,1,&good,1,1,s,&c)==CE_PRODUCT2_INVALID_INDEX,"out of bounds topology accepted");
 require(ce_product2_cuda_create(3,1,&good,UINT64_MAX,&good,1,1,s,&c)==CE_PRODUCT2_OVERFLOW,"declared topology byte overflow accepted");
 auto wrapped=reinterpret_cast<const int64_t*>(UINTPTR_MAX-7);
 require(ce_product2_cuda_create(3,2,wrapped,2,&good,2,1,s,&c)==CE_PRODUCT2_OVERFLOW,"topology end wrap accepted");
 ok(cudaStreamDestroy(s));std::puts("product2 native CUDA forward/input-VJP/coefficient-VJP/combined-JVP PASS; counts 0,1,5,129; repeated inputs and zeros; direct and prepared CUDA stage admission PASS");return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"product2 CUDA smoke FAIL: %s\n",e.what());return 1;}}
