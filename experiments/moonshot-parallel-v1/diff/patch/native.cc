#include <cmath>
#include <cstddef>
#include <vector>
#include <initializer_list>
namespace {
// Private ABI: callers must supply valid n*n storage. Raw pointers do not
// convey allocated extents; the validated Python wrapper owns that check.
bool valid(int n, std::initializer_list<const float*> pointers) {
 if(n <= 0 || n > 46340) return false;
 for(auto p : pointers) if(!p) return false;
 return true;
}
void mm(const float* a,const float* b,float* c,int n,bool at=false,bool bt=false) {
 for(int i=0;i<n;++i) for(int j=0;j<n;++j) {
  float s=0; for(int k=0;k<n;++k) s+=(at?a[k*n+i]:a[i*n+k])*(bt?b[j*n+k]:b[k*n+j]);
  c[i*n+j]=s;
 }
}
}
extern "C" {
int patch_output(int n,const float* v,const float* r,float* y) { if(!valid(n,{v,r,y})) return 1; mm(v,r,y,n); return 0; }
int patch_forward(int n,const float* x,const float* l,const float* r,float* t,float* v,float* y) {
 if(!valid(n,{x,l,r,t,v,y})) return 1;
 mm(l,x,t,n); for(int i=0;i<n*n;++i) v[i]=std::tanh(t[i]); mm(v,r,y,n); return 0;
}
int patch_vjp(int n,const float* x,const float* l,const float* r,const float* t,const float* v,const float* g,float* dx,float* dl,float* dr) {
 if(!valid(n,{x,l,r,t,v,g,dx,dl,dr})) return 1;
 try {
 std::vector<float> h(static_cast<std::size_t>(n)*n); mm(g,r,h.data(),n,false,true);
 for(int i=0;i<n*n;++i) { float a=std::tanh(t[i]); h[i]*=1-a*a; }
 mm(l,h.data(),dx,n,true,false); mm(h.data(),x,dl,n,false,true); mm(v,g,dr,n,true,false); return 0;
 } catch(...) { return 2; }
}
int patch_jvp(int n,const float* x,const float* l,const float* r,const float* t,const float* v,const float* dx,const float* dl,const float* dr,float* dy) {
 if(!valid(n,{x,l,r,t,v,dx,dl,dr,dy})) return 1;
 try {
 std::vector<float> a(static_cast<std::size_t>(n)*n),b(a.size()),c(a.size());
 mm(dl,x,a.data(),n); mm(l,dx,b.data(),n);
 for(int i=0;i<n*n;++i) { float h=std::tanh(t[i]); a[i]=(a[i]+b[i])*(1-h*h); }
 mm(a.data(),r,dy,n); mm(v,dr,c.data(),n); for(int i=0;i<n*n;++i) dy[i]+=c[i]; return 0;
 } catch(...) { return 2; }
}
}
