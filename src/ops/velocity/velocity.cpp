module;
#include <stddef.h>
#include <stdint.h>
#include <Cellerator/ops/numeric/binary16.hh>
module cellerator.velocity;

namespace cellerator::velocity {
namespace {
using anndatapp::DType;
using anndatapp::Layout;
using velocitypp::Code;
using velocitypp::Status;
struct Source {
    Layout layout; DType type, index_type, offset_type;
    const unsigned char *values, *indices, *offsets;
    size_t rows, columns, nnz; ptrdiff_t rs, cs; bool half;
};
template<class T> T load(const unsigned char* p) { T v; __builtin_memcpy(&v,p,sizeof v); return v; }
size_t width(DType t) {
    switch(t) {
    case DType::int8: case DType::uint8: return 1;
    case DType::int16: case DType::uint16: return 2;
    case DType::int32: case DType::uint32: case DType::float32: return 4;
    case DType::int64: case DType::uint64: case DType::float64: return 8;
    default: return 0;
    }
}
bool integer(DType t) { return t!=DType::float32 && t!=DType::float64 && width(t); }
template<class T> bool index_value(const unsigned char* p,size_t& v) {
    T x=load<T>(p); if(x<0)return false; v=size_t(x); return uint64_t(v)==uint64_t(x);
}
bool index(const unsigned char* p,DType t,size_t& v) {
    switch(t) {
    case DType::int8: return index_value<int8_t>(p,v); case DType::uint8: return index_value<uint8_t>(p,v);
    case DType::int16: return index_value<int16_t>(p,v); case DType::uint16: return index_value<uint16_t>(p,v);
    case DType::int32: return index_value<int32_t>(p,v); case DType::uint32: return index_value<uint32_t>(p,v);
    case DType::int64: return index_value<int64_t>(p,v); case DType::uint64: return index_value<uint64_t>(p,v);
    default: return false;
    }
}
size_t idx(const unsigned char* p,DType t,size_t i) { size_t v=0; index(p+i*width(t),t,v); return v; }
Status source(Matrix m,Source& s) {
    if(bool(m.half)==bool(m.acquired))return {Code::argument};
    if(m.half) {
        auto& h=*m.half; s={h.layout,DType::float32,h.index_type,h.offset_type,
            reinterpret_cast<const unsigned char*>(h.values),reinterpret_cast<const unsigned char*>(h.indices),
            reinterpret_cast<const unsigned char*>(h.offsets),h.rows,h.columns,h.nnz,h.row_stride_bytes,h.column_stride_bytes,true};
        return {};
    }
    auto& a=*m.acquired;
    if(a.dtype!=DType::float32 && a.dtype!=DType::float64)return {Code::dtype};
    if(a.layout==Layout::dense) {
        if(!a.dense || a.sparse || a.dense->rank()!=2)return {Code::argument};
        auto& d=*a.dense; if(d.element_size()!=width(a.dtype))return {Code::dtype};
        s={a.layout,a.dtype,a.index_type,a.offset_type,reinterpret_cast<const unsigned char*>(d.data()),nullptr,nullptr,
            d.extent(0),d.extent(1),0,d.stride_bytes(0),d.stride_bytes(1),false};
    } else {
        if(!a.sparse || a.dense)return {Code::argument};
        auto& d=*a.sparse;
        if(d.value_size()!=width(a.dtype) || d.index_size()!=width(a.index_type) || d.offset_size()!=width(a.offset_type))return {Code::dtype};
        s={a.layout,a.dtype,a.index_type,a.offset_type,reinterpret_cast<const unsigned char*>(d.values()),
            reinterpret_cast<const unsigned char*>(d.indices()),reinterpret_cast<const unsigned char*>(d.offsets()),
            d.rows(),d.columns(),d.nnz(),0,0,false};
    }
    return {};
}
template<class T> T value(const Source& a,const unsigned char* p) {
    return a.half ? T(numeric::load_f16(load<uint16_t>(p))) :
        a.type==DType::float32 ? T(load<float>(p)) : T(load<double>(p));
}
size_t value_width(const Source& a) { return a.half ? 2 : width(a.type); }
Status check(const Source& a) {
    if(a.rows>size_t(PTRDIFF_MAX) || a.columns>size_t(PTRDIFF_MAX))return {Code::overflow};
    if(a.layout==Layout::dense) {
        if(a.rows && a.columns && !a.values)return {Code::argument};
        ptrdiff_t r=0,c=0,lo=0,hi=0,end=0;
        if(a.rows && a.columns && (__builtin_mul_overflow(ptrdiff_t(a.rows-1),a.rs,&r) ||
           __builtin_mul_overflow(ptrdiff_t(a.columns-1),a.cs,&c) ||
           __builtin_add_overflow(r<0?r:0,c<0?c:0,&lo) ||
           __builtin_add_overflow(r>0?r:0,c>0?c:0,&hi) ||
           __builtin_add_overflow(hi,ptrdiff_t(value_width(a)-1),&end)))return {Code::overflow};
        for(size_t i=0;i<a.rows;++i)for(size_t j=0;j<a.columns;++j)
            if(!__builtin_isfinite(value<double>(a,a.values+ptrdiff_t(i)*a.rs+ptrdiff_t(j)*a.cs)))return {Code::nonfinite,i};
        return {};
    }
    if(a.layout!=Layout::csr && a.layout!=Layout::csc)return {Code::layout};
    if(!integer(a.index_type) || !integer(a.offset_type))return {Code::dtype};
    size_t major=a.layout==Layout::csr?a.rows:a.columns, minor=a.layout==Layout::csr?a.columns:a.rows;
    if(major==SIZE_MAX || major+1>size_t(PTRDIFF_MAX)/width(a.offset_type) ||
       a.nnz>size_t(PTRDIFF_MAX)/width(a.index_type) || a.nnz>size_t(PTRDIFF_MAX)/value_width(a))return {Code::overflow};
    if(!a.offsets || (a.nnz && (!a.values || !a.indices)))return {Code::argument};
    size_t prev=0,cur=0;
    if(!index(a.offsets,a.offset_type,prev) || prev)return {Code::bounds};
    for(size_t i=0;i<major;++i) {
        if(!index(a.offsets+(i+1)*width(a.offset_type),a.offset_type,cur) || cur<prev || cur>a.nnz)return {Code::bounds,i};
        for(size_t p=prev;p<cur;++p) {
            size_t j=0;
            if(!index(a.indices+p*width(a.index_type),a.index_type,j) || j>=minor)return {Code::bounds,p};
            if(!__builtin_isfinite(value<double>(a,a.values+p*value_width(a))))return {Code::nonfinite,p};
        }
        prev=cur;
    }
    return prev==a.nnz ? Status{} : Status{Code::bounds,major};
}
// Gather one logical column; duplicates are added in original stored order.
template<class T> void column(const Source& a,size_t j,T* out) {
    for(size_t i=0;i<a.rows;++i)out[i]=0;
    if(a.layout==Layout::dense) {
        for(size_t i=0;i<a.rows;++i)out[i]=value<T>(a,a.values+ptrdiff_t(i)*a.rs+ptrdiff_t(j)*a.cs);
    } else if(a.layout==Layout::csc) {
        for(size_t p=idx(a.offsets,a.offset_type,j);p<idx(a.offsets,a.offset_type,j+1);++p)
            out[idx(a.indices,a.index_type,p)]+=value<T>(a,a.values+p*value_width(a));
    } else for(size_t i=0;i<a.rows;++i)
        for(size_t p=idx(a.offsets,a.offset_type,i);p<idx(a.offsets,a.offset_type,i+1);++p)
            if(idx(a.indices,a.index_type,p)==j)out[i]+=value<T>(a,a.values+p*value_width(a));
}
template<class T> void add(T w,T x,T y,size_t q,T* ms,T* mu,T* mss,T* mus,T* muu) {
    if(ms)ms[q]+=w*x; if(mu)mu[q]+=w*y;
    if(mss)mss[q]+=w*(x*x); if(mus)mus[q]+=w*(x*y); if(muu)muu[q]+=w*(y*y);
}
// Traverse W by source column; this also preserves each duplicate edge.
template<class T> void apply(const Source& w,const T* s,const T* u,size_t n,size_t stride,
    T* ms,T* mu,T* mss,T* mus,T* muu) {
    for(size_t i=0;i<n;++i) { size_t q=i*stride; if(ms)ms[q]=0; if(mu)mu[q]=0; if(mss)mss[q]=0; if(mus)mus[q]=0; if(muu)muu[q]=0; }
    if(w.layout==Layout::csr) {
        for(size_t i=0;i<n;++i)for(size_t p=idx(w.offsets,w.offset_type,i);p<idx(w.offsets,w.offset_type,i+1);++p) {
            size_t k=idx(w.indices,w.index_type,p);
            add(value<T>(w,w.values+p*value_width(w)),s[k],u[k],i*stride,ms,mu,mss,mus,muu);
        }
        return;
    }
    for(size_t k=0;k<n;++k) {
        T x=s[k],y=u[k];
        if(w.layout==Layout::dense) for(size_t i=0;i<n;++i)
            add(value<T>(w,w.values+ptrdiff_t(i)*w.rs+ptrdiff_t(k)*w.cs),x,y,i*stride,ms,mu,mss,mus,muu);
        else if(w.layout==Layout::csc) for(size_t p=idx(w.offsets,w.offset_type,k);p<idx(w.offsets,w.offset_type,k+1);++p)
            add(value<T>(w,w.values+p*value_width(w)),x,y,idx(w.indices,w.index_type,p)*stride,ms,mu,mss,mus,muu);
    }
}
template<class T> void calculate(const Input& in,size_t first,size_t count,T* ms,T* mu,T* mss,T* mus,T* muu,T* s,T* u) {
    Source a{},b{},w{}; source(in.spliced,a);source(in.unspliced,b);source(in.weights,w);
    for(size_t j=0;j<count;++j) {
        size_t gene=in.domain.gene_indices?in.domain.gene_indices[first+j]:first+j;
        column(a,gene,s);column(b,gene,u);
        apply(w,s,u,in.domain.rows,count,ms?ms+j:nullptr,mu?mu+j:nullptr,mss?mss+j:nullptr,mus?mus+j:nullptr,muu?muu+j:nullptr);
    }
}
}
Status validate(const Input& in) {
    auto e=velocitypp::validate(in.domain);if(!e)return e;
    Source s{},u{},w{};e=source(in.spliced,s);if(!e)return e;e=source(in.unspliced,u);if(!e)return e;e=source(in.weights,w);if(!e)return e;
    if(s.rows!=in.domain.rows || u.rows!=s.rows || s.columns!=in.domain.source_genes || u.columns!=s.columns || w.rows!=s.rows || w.columns!=s.rows)return {Code::shape};
    e=check(s);if(!e)return e;e=check(u);return e?check(w):e;
}
template<class T> void moments(const Input& in,size_t first,size_t count,T* ms,T* mu,T* s,T* u) {
    calculate(in,first,count,ms,mu,(T*)nullptr,(T*)nullptr,(T*)nullptr,s,u);
}
template<class T> void second_moments(const Input& in,size_t first,size_t count,T* ms,T* mu,T* mss,T* mus,T* muu,T* s,T* u) {
    calculate(in,first,count,ms,mu,mss,mus,muu,s,u);
}
template<class T> void residuals_f16(const velocitypp::BasicMomentBlock<T>& m,const velocitypp::BasicFitBlock<T>& f,T* scratch,uint16_t* out,ptrdiff_t rs,ptrdiff_t cs) {
    velocitypp::reference::residuals(m,f,scratch,ptrdiff_t(m.count));
    for(size_t i=0;i<m.rows;++i)for(size_t j=0;j<m.count;++j)out[ptrdiff_t(i)*rs+ptrdiff_t(j)*cs]=numeric::round_f16(scratch[i*m.count+j]);
}
template<class T> void stochastic_residuals_f16(const velocitypp::BasicStochasticMomentBlock<T>& m,const velocitypp::BasicStochasticFitBlock<T>& f,T* scratch,uint16_t* out,uint16_t* second,ptrdiff_t rs,ptrdiff_t cs) {
    if(out) { velocitypp::reference::stochastic_residuals(m,f,scratch,(T*)nullptr,ptrdiff_t(m.count));
        for(size_t i=0;i<m.rows;++i)for(size_t j=0;j<m.count;++j)out[ptrdiff_t(i)*rs+ptrdiff_t(j)*cs]=numeric::round_f16(scratch[i*m.count+j]); }
    if(second) { velocitypp::reference::stochastic_residuals(m,f,(T*)nullptr,scratch,ptrdiff_t(m.count));
        for(size_t i=0;i<m.rows;++i)for(size_t j=0;j<m.count;++j)second[ptrdiff_t(i)*rs+ptrdiff_t(j)*cs]=numeric::round_f16(scratch[i*m.count+j]); }
}
namespace {
template<class T> velocitypp::precision::PrecisionCode replace(void* context,size_t position,size_t offset,
    const velocitypp::precision::OriginalColumn& c,bool second) {
    using velocitypp::precision::PrecisionCode;
    if(!context)return PrecisionCode::invalid_argument;
    auto& p=*static_cast<PromotionMoments<T>*>(context);
    if(!p.input || c.rows!=p.input->domain.rows || (c.rows && (!c.spliced || !c.unspliced || !p.source_s || !p.source_u)) ||
       (c.spliced_type!=DType::float32 && c.spliced_type!=DType::float64) ||
       (c.unspliced_type!=DType::float32 && c.unspliced_type!=DType::float64))return PrecisionCode::invalid_argument;
    auto* x=reinterpret_cast<const unsigned char*>(c.spliced);
    auto* y=reinterpret_cast<const unsigned char*>(c.unspliced);
    for(size_t i=0;i<c.rows;++i) {
        const auto* sx=x+ptrdiff_t(i)*c.spliced_stride_bytes;
        const auto* uy=y+ptrdiff_t(i)*c.unspliced_stride_bytes;
        p.source_s[i]=c.spliced_type==DType::float32?T(load<float>(sx)):T(load<double>(sx));
        p.source_u[i]=c.unspliced_type==DType::float32?T(load<float>(uy)):T(load<double>(uy));
    }
    Source w{};if(!source(p.input->weights,w))return PrecisionCode::invalid_argument;
    if(!second) {
        auto* b=p.initial;
        if(!b || b->rows!=c.rows || b->stride<=0 || offset>=b->count || position<b->first || position-b->first!=offset ||
           (c.rows && (!b->spliced || !b->unspliced)))return PrecisionCode::invalid_argument;
        if(c.rows)apply(w,p.source_s,p.source_u,c.rows,size_t(b->stride),b->spliced+offset,b->unspliced+offset,(T*)nullptr,(T*)nullptr,(T*)nullptr);
    } else {
        auto* b=p.second;
        if(!b || b->rows!=c.rows || b->stride<=0 || offset>=b->count || position<b->first || position-b->first!=offset ||
           (c.rows && (!b->spliced || !b->unspliced || !b->spliced_second || !b->cross)))return PrecisionCode::invalid_argument;
        if(c.rows)apply(w,p.source_s,p.source_u,c.rows,size_t(b->stride),b->spliced+offset,b->unspliced+offset,
            b->spliced_second+offset,b->cross+offset,b->unspliced_second?b->unspliced_second+offset:nullptr);
    }
    return PrecisionCode::ok;
}
}
velocitypp::precision::PrecisionCode replace32(void* p,size_t k,size_t j,const velocitypp::precision::OriginalColumn& c,bool second) {return replace<float>(p,k,j,c,second);}
velocitypp::precision::PrecisionCode replace64(void* p,size_t k,size_t j,const velocitypp::precision::OriginalColumn& c,bool second) {return replace<double>(p,k,j,c,second);}
#define CELLERATOR_VELOCITY(T) \
template void moments<T>(const Input&,size_t,size_t,T*,T*,T*,T*); \
template void second_moments<T>(const Input&,size_t,size_t,T*,T*,T*,T*,T*,T*,T*); \
template void residuals_f16<T>(const velocitypp::BasicMomentBlock<T>&,const velocitypp::BasicFitBlock<T>&,T*,uint16_t*,ptrdiff_t,ptrdiff_t); \
template void stochastic_residuals_f16<T>(const velocitypp::BasicStochasticMomentBlock<T>&,const velocitypp::BasicStochasticFitBlock<T>&,T*,uint16_t*,uint16_t*,ptrdiff_t,ptrdiff_t);
CELLERATOR_VELOCITY(float)
CELLERATOR_VELOCITY(double)
#undef CELLERATOR_VELOCITY
}
