#include "oracle.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <initializer_list>
using cellerator::experimental::moonshot::product::oracle;
void check(bool b) { if(!b) { std::fprintf(stderr,"product oracle failed\n");std::exit(1); } }
int main() {
    const float x[]={0,3,-2,0.5f},v[]={2,-1,4,-3};
    for(unsigned n:{0u,1u,31u,32u,33u,127u,129u}) {
        for(unsigned i=0;i<n;++i) {
            unsigned a=i%4,b=(i%3==0)?a:(i+1)%4;
            float y,dy,k=float(int(i%5)-2)*0.3f;
            oracle(x[a],x[b],v[a],v[b],k,y,dy);
            double ey=double(k)*x[a]*x[b];
            const double eps=1e-4;
            double plus=double(k)*(x[a]+eps*v[a])*(x[b]+eps*v[b]);
            double minus=double(k)*(x[a]-eps*v[a])*(x[b]-eps*v[b]);
            check(std::abs(double(y)-ey)<2e-6*(1+std::abs(ey)));
            check(std::abs(double(dy)-(plus-minus)/(2*eps))<2e-6*(1+std::abs(dy)));
        }
    }
    float y,dy;oracle(0,3,2,-1,5,y,dy);check(y==0 && dy==30);
    oracle(0,0,2,3,5,y,dy);check(y==0 && dy==0);
    oracle(3,3,-1,-1,2,y,dy);check(y==18 && dy==-12);
    std::puts("product FP32 oracle counts 0/1/31/32/33/127/129 PASS");
}
