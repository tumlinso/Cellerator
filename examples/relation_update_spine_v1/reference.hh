#pragma once
// Demo/reference mathematics only. No Cellerator implementation or GPU fallback.
#include <array>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
namespace ru1_demo {
inline constexpr std::size_t sources=20, destinations=19, width=16, edges=262;
inline void require(bool value,const std::string& message) {
    if(!value)throw std::runtime_error(message);
}
inline std::uint16_t to_half(float value) {
    const auto b=std::bit_cast<std::uint32_t>(value);
    const auto sign=std::uint16_t((b>>16)&0x8000);
    const unsigned exp=(b>>23)&255, mant=b&0x7fffff;
    if(exp==255)return std::uint16_t(sign|0x7c00|(mant?0x200:0));
    const int e=int(exp)-127+15;
    if(e>=31)return std::uint16_t(sign|0x7c00);
    if(e<=0) {
        if(e< -10)return sign;
        const unsigned m=mant|0x800000, shift=unsigned(14-e);
        unsigned q=m>>shift;
        const unsigned r=m&((1u<<shift)-1), halfway=1u<<(shift-1);
        if(r>halfway||(r==halfway&&(q&1)))++q;
        return std::uint16_t(sign|q);
    }
    unsigned q=mant>>13;
    const unsigned r=mant&8191;
    if(r>4096||(r==4096&&(q&1)))++q;
    return std::uint16_t(sign|((unsigned(e)<<10)+q));
}
inline float from_half(std::uint16_t h) {
    const int e=(h>>10)&31,m=h&1023;
    float x;
    if(e==31)x=m?std::numeric_limits<float>::quiet_NaN():std::numeric_limits<float>::infinity();
    else if(e==0)x=std::ldexp(float(m),-24);
    else x=std::ldexp(float(1024+m),e-25);
    return (h&0x8000)?-x:x;
}
inline float qhalf(float x){return from_half(to_half(x));}
struct fixture {
    std::array<std::uint32_t,destinations+1> offsets{};
    std::array<std::uint32_t,edges> source{},destination{};
    std::array<std::uint16_t,edges> weights{};
    std::array<float,sources*width> x{};
    std::array<float,destinations*width> target{};
    fixture() {
        std::size_t e=0;
        for(std::size_t d=0;d<destinations;++d) {
            offsets[d]=std::uint32_t(e);
            auto add=[&](std::uint32_t s){source[e]=s;destination[e]=std::uint32_t(d);++e;};
            if(d<16)for(std::uint32_t s=0;s<16;++s)add(s);
            else if(d==16){add(0);add(3);add(16);}
            else if(d==17){add(2);add(16);add(18);}
        }
        offsets[destinations]=std::uint32_t(e);require(e==edges,"fixture edge count");
        for(std::size_t s=0;s<sources;++s)for(std::size_t k=0;k<width;++k)
            x[s*width+k]=float(int((s*7+k*5)%17)-8)/16.0f;
        for(std::size_t i=0;i<edges;++i) {
            const float v=float(int((source[i]*3+destination[i]*5)%13)-6)/64.0f;
            weights[i]=to_half(v);
            const float teacher=qhalf(v+float(int(i%5)-2)/128.0f);
            for(std::size_t k=0;k<width;++k)
                target[destination[i]*width+k]+=teacher*x[source[i]*width+k];
        }
        // Bounded non-half-representable target perturbation exercises explicit VJP rounding.
        for(std::size_t i=0;i<18*width;++i)target[i]+=float(int(i%7)-3)*0.00017f;
    }
};
inline std::array<double,destinations*width> forward(const fixture& f,const std::array<std::uint16_t,edges>& w) {
    std::array<double,destinations*width> y{};
    for(std::size_t e=0;e<edges;++e)for(std::size_t k=0;k<width;++k)
        y[f.destination[e]*width+k]+=double(from_half(w[e]))*f.x[f.source[e]*width+k];
    return y;
}
inline std::array<double,sources*width> transpose(const fixture& f,const std::array<std::uint16_t,edges>& w,const std::array<float,destinations*width>& dy) {
    std::array<double,sources*width> dx{};
    for(std::size_t e=0;e<edges;++e)for(std::size_t k=0;k<width;++k)
        dx[f.source[e]*width+k]+=double(from_half(w[e]))*dy[f.destination[e]*width+k];
    return dx;
}
inline std::array<double,edges> gradient(const fixture& f,const std::array<float,destinations*width>& dy,bool half_operands) {
    std::array<double,edges> g{};
    for(std::size_t e=0;e<edges;++e)for(std::size_t k=0;k<width;++k) {
        const float x=f.x[f.source[e]*width+k],v=dy[f.destination[e]*width+k];
        g[e]+=double(half_operands?qhalf(x):x)*(half_operands?qhalf(v):v);
    }
    return g;
}
template<class T> inline double loss(const fixture& f,const std::array<T,destinations*width>& y) {
    double l=0;for(std::size_t i=0;i<y.size();++i){const double r=double(y[i])-f.target[i];l+=r*r;}
    return l/(2.0*width);
}
template<class T> inline std::array<float,destinations*width> cotangent(const fixture& f,const std::array<T,destinations*width>& y) {
    std::array<float,destinations*width> dy{};
    for(std::size_t i=0;i<dy.size();++i)dy[i]=float((double(y[i])-f.target[i])/width);
    return dy;
}
inline void half_tests() {
    require(to_half(1.0f)==0x3c00,"half one");
    require(to_half(1.0f+std::ldexp(1.0f,-11))==0x3c00,"half even tie");
    require(to_half(1.0f+3*std::ldexp(1.0f,-11))==0x3c02,"half odd tie");
    require(to_half(std::ldexp(1.0f,-24))==1,"half subnormal");
    require(to_half(std::ldexp(1.0f,-25))==0,"half underflow tie");
    require(to_half(-0.0f)==0x8000,"negative zero");
    require(to_half(65504.0f)==0x7bff,"half max");
    require(to_half(65520.0f)==0x7c00,"half overflow tie");
    for(unsigned i=0;i<65536;++i)if((i&0x7c00)!=0x7c00)
        require(to_half(from_half(std::uint16_t(i)))==i,"all finite half roundtrips");
}
struct reference_result {double loss0,loss1,loss2,max_fd_error,max_mixed_error;};
inline reference_result reference_check() {
    half_tests();fixture f;
    auto y=forward(f,f.weights);auto dy=cotangent(f,y);auto g=gradient(f,dy,false);auto gh=gradient(f,dy,true);
    const auto dx=transpose(f,f.weights,dy);
    double lhs=0,rhs=0;for(std::size_t i=0;i<y.size();++i)lhs+=y[i]*dy[i];
    for(std::size_t i=0;i<dx.size();++i)rhs+=dx[i]*f.x[i];
    require(std::abs(lhs-rhs)<1e-11,"adjoint identity");
    // Finite differences on continuous weights, not through binary16 rounding.
    double fd_error=0,mixed=0;
    for(std::size_t e=0;e<edges;++e) {
        constexpr double h=1e-5;auto plus=y,minus=y;
        for(std::size_t k=0;k<width;++k){const auto i=f.destination[e]*width+k;const double v=h*f.x[f.source[e]*width+k];plus[i]+=v;minus[i]-=v;}
        const double fd=(loss(f,plus)-loss(f,minus))/(2*h);
        fd_error=std::max(fd_error,std::abs(fd-g[e]));mixed=std::max(mixed,std::abs(g[e]-gh[e]));
    }
    require(fd_error<1e-8,"finite difference VJP");
    auto w1=f.weights;constexpr float alpha=0.125f;
    for(std::size_t e=0;e<edges;++e)w1[e]=to_half(from_half(w1[e])+(-alpha*float(gh[e])));
    const auto y1=forward(f,w1);auto gh1=gradient(f,cotangent(f,y1),true);auto w2=w1;
    for(std::size_t e=0;e<edges;++e)w2[e]=to_half(std::fma(-alpha,float(gh1[e]),from_half(w1[e])));
    const auto y2=forward(f,w2);
    const double l0=loss(f,y),l1=loss(f,y1),l2=loss(f,y2);
    require(l1<l0&&l2<l1,"bounded fixture loss improvement");
    require(y[18*width]==0&&dx[19*width]==0,"empty row/isolated source");
    return {l0,l1,l2,fd_error,mixed};
}
} // namespace ru1_demo
