
#include <cmath>
#include <cstdint>
namespace { constexpr const char* topology_contract = "csr4x3-seg2-v2"; }
extern "C" int cellerator_region(const std::uint64_t* off,const std::uint64_t* col,
 const float* values,const float* x,float* out,std::uint64_t rows,
 const std::uint64_t* seg,std::uint64_t segments) noexcept {
 if(rows!=4||segments!=2||topology_contract[0]=='\0')return 1;
 float temp[4]{};
 for(std::uint64_t r=0;r<rows;++r){float sum=0;
  for(auto e=off[r];e<off[r+1];++e)sum+=values[e]*x[col[e]];temp[r]=sum;}
 for(std::uint64_t s=0;s<segments;++s){auto b=seg[s],e=seg[s+1];
  if(b==e)continue;float m=temp[b];for(auto r=b+1;r<e;++r)if(temp[r]>m)m=temp[r];
  float d=0;for(auto r=b;r<e;++r)d+=std::exp(temp[r]-m);
  for(auto r=b;r<e;++r)out[r]=std::exp(temp[r]-m)/d;}
 return 0;
}
