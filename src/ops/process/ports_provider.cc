#include <cstdint>
#include <vector>
#include <algorithm>
#include <limits>
// Private ctypes ABI: buffer lengths are established by the Python Tape constructor.
// Raw callers must supply arrays with those lengths; arbitrary pointer extents cannot
// be proven here. Independently reject invalid scalar/shape/index metadata.
// Private actor row-major E[P,H_i] and D[H_i,P]; one weight per edge.
extern "C" int ports_transport(int mode, int64_t n, int64_t p, int64_t k,
 const int64_t* widths, const int64_t* src, const int64_t* dst,
 const float* h, const float* e, const float* d, const float* w,
 const float* a, const float* b, const float* c, const float* f,
 float* out, float* oe, float* od, float* ow) noexcept {
 try {
  if(mode<0 || mode>2 || n<=0 || p<=0 || k<0 || !widths || !h || !e || !d || !out) return 2;
  if(k && (!src || !dst || !w)) return 2;
  if(mode==1 && (!a || !oe || !od || (k && !ow))) return 2;
  if(mode==2 && (!a || !b || !c || (k && !f))) return 2;
  const int64_t limit=std::numeric_limits<int64_t>::max()/sizeof(float);
  if(n>limit/p) return 2;
  int64_t checked_total=0;
  for(int64_t i=0;i<n;++i){
   if(widths[i]<=0 || widths[i]>limit-checked_total) return 2;
   checked_total+=widths[i];
  }
  if(checked_total>limit/p) return 2;
  for(int64_t t=0;t<k;++t) if(src[t]<0 || src[t]>=n || dst[t]<0 || dst[t]>=n) return 2;
  std::vector<int64_t> off(n+1); for(int64_t i=0;i<n;++i) off[i+1]=off[i]+widths[i];
  const int64_t total=off[n];
  std::vector<float> m(n*p,0),z(n*p,0),dm(n*p,0),dz(n*p,0);
  for(int64_t i=0;i<n;++i) for(int64_t r=0;r<p;++r) for(int64_t j=0;j<widths[i];++j){
   const int64_t ei=p*off[i]+r*widths[i]+j;
   m[i*p+r]+=e[ei]*h[off[i]+j];
   if(mode==2) dm[i*p+r]+=b[ei]*h[off[i]+j]+e[ei]*a[off[i]+j];
  }
  for(int64_t t=0;t<k;++t) for(int64_t r=0;r<p;++r){
   z[dst[t]*p+r]+=w[t]*m[src[t]*p+r];
   if(mode==2) dz[dst[t]*p+r]+=f[t]*m[src[t]*p+r]+w[t]*dm[src[t]*p+r];
  }
  std::fill(out,out+total,0);
  if(mode!=1){
   for(int64_t i=0;i<n;++i) for(int64_t j=0;j<widths[i];++j) for(int64_t r=0;r<p;++r){
    const int64_t di=(off[i]+j)*p+r;
    out[off[i]+j]+=mode==0 ? d[di]*z[i*p+r] : c[di]*z[i*p+r]+d[di]*dz[i*p+r];
   }
  } else {
   std::fill(oe,oe+p*total,0); std::fill(od,od+p*total,0); if(k) std::fill(ow,ow+k,0);
   for(int64_t i=0;i<n;++i) for(int64_t j=0;j<widths[i];++j) for(int64_t r=0;r<p;++r){
    dz[i*p+r]+=d[(off[i]+j)*p+r]*a[off[i]+j];
    od[(off[i]+j)*p+r]=a[off[i]+j]*z[i*p+r];
   }
   for(int64_t t=0;t<k;++t) for(int64_t r=0;r<p;++r){
    dm[src[t]*p+r]+=w[t]*dz[dst[t]*p+r];
    ow[t]+=dz[dst[t]*p+r]*m[src[t]*p+r];
   }
   for(int64_t i=0;i<n;++i) for(int64_t r=0;r<p;++r) for(int64_t j=0;j<widths[i];++j){
    const int64_t ei=p*off[i]+r*widths[i]+j;
    out[off[i]+j]+=e[ei]*dm[i*p+r]; oe[ei]=dm[i*p+r]*h[off[i]+j];
   }
  }
  return 0;
 } catch (...) {return 1;}
}
