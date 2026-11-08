#include "bridge.hh"
#include "../detail/scientific.hh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <limits>

namespace velocity_cuda_bridge {
namespace sci = velocity::detail::scientific;
using anndata::device::SparseLayout;
namespace {
constexpr unsigned threads = 128;
unsigned blocks(std::uint64_t n) { return unsigned((n + threads - 1)/threads > 65535 ? 65535 : (n + threads - 1)/threads); }
cudaStream_t native(void* stream) { return reinterpret_cast<cudaStream_t>(stream); }

__host__ __device__ unsigned width(ScalarType t) {
    return t == ScalarType::boolean || t == ScalarType::int8 || t == ScalarType::uint8 ? 1
         : t == ScalarType::int16 || t == ScalarType::uint16 || t == ScalarType::float16 ? 2
         : t == ScalarType::int32 || t == ScalarType::uint32 || t == ScalarType::float32 ? 4 : 8;
}
__device__ std::uint64_t integer(const void* p, ScalarType t, std::uint64_t i) {
    switch (t) {
    case ScalarType::int8: return reinterpret_cast<const std::int8_t*>(p)[i];
    case ScalarType::uint8: return reinterpret_cast<const std::uint8_t*>(p)[i];
    case ScalarType::int16: return reinterpret_cast<const std::int16_t*>(p)[i];
    case ScalarType::uint16: return reinterpret_cast<const std::uint16_t*>(p)[i];
    case ScalarType::int32: return reinterpret_cast<const std::int32_t*>(p)[i];
    case ScalarType::uint32: return reinterpret_cast<const std::uint32_t*>(p)[i];
    case ScalarType::int64: return reinterpret_cast<const std::int64_t*>(p)[i];
    default: return reinterpret_cast<const std::uint64_t*>(p)[i];
    }
}
template<class T> __device__ T load(const void* p, ScalarType dtype, std::uint64_t i) {
    return dtype == ScalarType::float16 ? T(__half2float(__ushort_as_half(reinterpret_cast<const unsigned short*>(p)[i])))
         : dtype == ScalarType::float32 ? T(reinterpret_cast<const float*>(p)[i])
         : T(reinterpret_cast<const double*>(p)[i]);
}
__device__ const char* address(DenseView v, std::uint64_t r, std::uint64_t c) {
    return static_cast<const char*>(v.values.data) + v.origin
        + (std::int64_t(r)*v.row_stride + std::int64_t(c)*v.column_stride)*std::int64_t(width(v.dtype));
}
template<class T> __device__ T dense(DenseView v, std::uint64_t r, std::uint64_t c) { return load<T>(address(v,r,c),v.dtype,0); }
template<class T> __device__ void store(DenseView v, std::uint64_t r, std::uint64_t c, T x) {
    char* p = const_cast<char*>(address(v,r,c));
    if (v.dtype == ScalarType::float16) {
        __half h;
        if constexpr (sizeof(T) == sizeof(double)) h = __double2half(x);
        else h = __float2half_rn(x);
        *reinterpret_cast<unsigned short*>(p) = __half_as_ushort(h);
    } else if (v.dtype == ScalarType::float32) *reinterpret_cast<float*>(p) = float(x);
    else *reinterpret_cast<double*>(p) = double(x);
}
__device__ bool enabled(DenseView mask, std::uint64_t i) {
    return !mask.values.data || *reinterpret_cast<const unsigned char*>(address(mask,i,0)) != 0;
}
__global__ void pack_mask(DenseView mask, std::uint64_t n, std::uint8_t* out) {
    for (std::uint64_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += std::uint64_t(gridDim.x)*blockDim.x)
        out[i] = enabled(mask,i);
}

__global__ void validate_values(Matrix m, int* error) {
    std::uint64_t n = m.sparse ? m.compressed.nnz : m.dense.rows*m.dense.columns;
    for (std::uint64_t p = blockIdx.x*blockDim.x + threadIdx.x; p < n; p += std::uint64_t(gridDim.x)*blockDim.x) {
        double x = m.sparse ? load<double>(m.compressed.values.data,m.compressed.dtype,p)
                           : dense<double>(m.dense,p/m.dense.columns,p%m.dense.columns);
        if (!isfinite(x)) atomicOr(error,4);
    }
}
__global__ void count_major(SparseView m, bool rows, std::uint64_t* counts) {
    auto major = m.layout == SparseLayout::csr ? m.rows : m.columns;
    for (std::uint64_t i = blockIdx.x*blockDim.x + threadIdx.x; i < major; i += std::uint64_t(gridDim.x)*blockDim.x)
        for (auto p = integer(m.offsets.data,m.offset_type,i), e = integer(m.offsets.data,m.offset_type,i+1); p < e; ++p) {
            auto index = integer(m.indices.data,m.index_type,p);
            auto r = m.layout == SparseLayout::csr ? i : index;
            auto c = m.layout == SparseLayout::csr ? index : i;
            atomicAdd(reinterpret_cast<unsigned long long*>(counts + (rows ? r : c)),1ull);
        }
}
__global__ void fill_edges(SparseView m, bool rows, const std::uint64_t* offsets,
                           std::uint64_t* cursors, Edge* edges) {
    auto major = m.layout == SparseLayout::csr ? m.rows : m.columns;
    for (std::uint64_t i = blockIdx.x*blockDim.x + threadIdx.x; i < major; i += std::uint64_t(gridDim.x)*blockDim.x)
        for (auto p = integer(m.offsets.data,m.offset_type,i), e = integer(m.offsets.data,m.offset_type,i+1); p < e; ++p) {
            auto index = integer(m.indices.data,m.index_type,p);
            auto r = m.layout == SparseLayout::csr ? i : index;
            auto c = m.layout == SparseLayout::csr ? index : i;
            auto target = rows ? r : c;
            auto q = offsets[target] + atomicAdd(reinterpret_cast<unsigned long long*>(cursors+target),1ull);
            edges[q] = {rows ? c : r,p};
        }
}
__device__ bool less(Edge a, Edge b) { return a.index < b.index || (a.index == b.index && a.position < b.position); }
__device__ void sift(Edge* a, std::uint64_t root, std::uint64_t n) {
    while (root < n/2) {
        auto child = 2*root+1;
        if (child+1<n && less(a[child],a[child+1])) ++child;
        if (!less(a[root],a[child])) return;
        auto t=a[root]; a[root]=a[child]; a[child]=t; root=child;
    }
}
__global__ void sort_edges(std::uint64_t n, const std::uint64_t* offsets, Edge* entries) {
    for (std::uint64_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += std::uint64_t(gridDim.x)*blockDim.x) {
        auto size=offsets[i+1]-offsets[i];if(size<2)continue;
        auto* a=entries+offsets[i];
        for(auto start=size/2;start>0;--start) sift(a,start-1,size);
        for(auto end=size;end>1;--end) {auto t=a[0];a[0]=a[end-1];a[end-1]=t;sift(a,0,end-1);}
    }
}

template<class T> __global__ void pack_dense(DenseView x, const std::uint64_t* genes,
    std::uint64_t first, const std::uint64_t* destination, std::uint64_t count, std::uint64_t rows, T* out) {
    for (std::uint64_t p = blockIdx.x*blockDim.x + threadIdx.x; p < rows*count; p += std::uint64_t(gridDim.x)*blockDim.x) {
        auto j=p/rows, r=p%rows, g=genes ? genes[first+j] : first+j;
        out[(destination ? destination[j] : j)*rows+r] = dense<T>(x,r,g);
    }
}
template<class T> __global__ void pack_sparse(PreparedMatrix x, const std::uint64_t* genes,
    std::uint64_t first, const std::uint64_t* destination, std::uint64_t count, std::uint64_t rows, T* out) {
    for (std::uint64_t j = blockIdx.x*blockDim.x + threadIdx.x; j < count; j += std::uint64_t(gridDim.x)*blockDim.x) {
        auto gene=genes ? genes[first+j] : first+j, column=destination ? destination[j] : j;
        if (destination) for(std::uint64_t r=0;r<rows;++r) out[column*rows+r]=0;
        for(auto p=x.offsets[gene];p<x.offsets[gene+1];++p) {
            auto e=x.edges[p]; out[column*rows+e.index] += load<T>(x.matrix.compressed.values.data,x.matrix.compressed.dtype,e.position);
        }
    }
}
template<class T> int pack_one(PreparedMatrix x, const std::uint64_t* genes, std::uint64_t first,
    const std::uint64_t* destination, std::uint64_t count, std::uint64_t rows, void* out, cudaStream_t stream) {
    if (!count || !rows) return 0;
    if (!x.matrix.sparse)
        pack_dense<T><<<blocks(rows*count),threads,0,stream>>>(x.matrix.dense,genes,first,destination,count,rows,static_cast<T*>(out));
    else {
        if (!destination) { auto e=cudaMemsetAsync(out,0,std::size_t(rows*count)*sizeof(T),stream); if(e) return e; }
        pack_sparse<T><<<blocks(count),threads,0,stream>>>(x,genes,first,destination,count,rows,static_cast<T*>(out));
    }
    return cudaGetLastError();
}

template<class T> __global__ void moment_fields(PreparedMatrix graph, std::uint64_t rows,
    std::uint64_t count, Fields f, bool second) {
    auto* s=static_cast<const T*>(f.source_s); auto* u=static_cast<const T*>(f.source_u);
    for(std::uint64_t p=blockIdx.x*blockDim.x+threadIdx.x;p<rows*count;p+=std::uint64_t(gridDim.x)*blockDim.x) {
        auto row=p%rows, gene=p/rows;
        T ms=0,mu=0,mss=0,mus=0,muu=0;
        auto begin=graph.matrix.sparse ? graph.offsets[row] : 0;
        auto end=graph.matrix.sparse ? graph.offsets[row+1] : rows;
        for(auto q=begin;q<end;++q) {
            auto source=graph.matrix.sparse ? graph.edges[q].index : q;
            T w=graph.matrix.sparse ? load<T>(graph.matrix.compressed.values.data,graph.matrix.compressed.dtype,graph.edges[q].position)
                                    : dense<T>(graph.matrix.dense,row,source);
            T x=s[gene*rows+source],y=u[gene*rows+source];
            ms+=w*x; mu+=w*y;
            if(second) {mss+=w*(x*x);mus+=w*(x*y);if(f.muu)muu+=w*(y*y);}
        }
        static_cast<T*>(f.ms)[p]=ms; static_cast<T*>(f.mu)[p]=mu;
        if(second) {static_cast<T*>(f.mss)[p]=mss;static_cast<T*>(f.mus)[p]=mus;if(f.muu)static_cast<T*>(f.muu)[p]=muu;}
    }
}
template<class T> __host__ __device__ sci::Options<T> options(Options o) {
    return {sci::Trimming(o.trimming),T(o.percentile_low),T(o.percentile_high),bool(o.fit_intercept),
        bool(o.positive_intercept),bool(o.r2_adjusted),T(o.ratio_min),T(o.ratio_max),T(o.min_r2),T(o.min_ratio)};
}
template<class T> __device__ sci::Fit<T> initial(Result r, std::uint64_t g) {
    return {static_cast<T*>(r.initial_gamma)[g],static_cast<T*>(r.initial_offset)[g],static_cast<T*>(r.r2)[g],
        sci::FitStatus(r.initial_status[g]),bool(r.selected[g])};
}
__device__ bool refine(Result r, std::uint64_t g, std::uint64_t gene, DenseView gene_mask) {
    return enabled(gene_mask,gene) && (r.selected[g] || *r.refit_all);
}
template<class T> __global__ void initialize(std::uint64_t rows, std::uint64_t first,
    std::uint64_t count, const std::uint64_t* genes, DenseView gene_mask, Fields f, Result r, Options o) {
    for(std::uint64_t j=blockIdx.x*blockDim.x+threadIdx.x;j<count;j+=std::uint64_t(gridDim.x)*blockDim.x) {
        auto g=first+j, gene=genes[g];
        auto fit=sci::fit_gene(static_cast<T*>(f.ms)+j*rows,static_cast<T*>(f.mu)+j*rows,rows,1,
            options<T>(o),static_cast<T*>(f.scores)+j*rows,f.mask,enabled(gene_mask,gene),sci::HeapSort{});
        static_cast<T*>(r.initial_gamma)[g]=fit.gamma;static_cast<T*>(r.initial_offset)[g]=fit.offset;
        static_cast<T*>(r.r2)[g]=fit.r2;r.selected[g]=fit.selected;r.initial_status[g]=std::uint8_t(fit.status);
        static_cast<T*>(r.gamma)[g]=fit.gamma;static_cast<T*>(r.offset)[g]=fit.offset;
    }
}

template<class T> __device__ unsigned long long key(T value) {
    if constexpr (sizeof(T)==4) {auto k=__float_as_uint(value);return k&0x80000000u ? ~k : k^0x80000000u;}
    else {auto k=static_cast<unsigned long long>(__double_as_longlong(value));return k&0x8000000000000000ull ? ~k : k^0x8000000000000000ull;}
}
template<class T> __device__ T key_value(unsigned long long k) {
    if constexpr (sizeof(T)==4) {auto u=std::uint32_t(k);return __uint_as_float(u&0x80000000u ? u^0x80000000u : ~u);}
    else {auto u=k&0x8000000000000000ull ? k^0x8000000000000000ull : ~k;return __longlong_as_double(static_cast<long long>(u));}
}
template<class T> __device__ T eligible_order_statistic(std::uint64_t n, const std::uint64_t* genes,
    DenseView mask, Result r, std::uint64_t rank) {
    // Rare global low-signal recovery: bounded radix selection, no G-sized
    // numerical scratch and no modification of the original R2 diagnostics.
    unsigned long long prefix=0, prefix_mask=0;
    for(int byte=int(sizeof(T))-1;byte>=0;--byte) {
        std::uint64_t bins[256]{};
        for(std::uint64_t g=0;g<n;++g) if(r.initial_status[g]==0 && enabled(mask,genes[g])) {
            auto k=key(static_cast<T*>(r.r2)[g]); if((k&prefix_mask)==prefix) ++bins[(k>>(8*byte))&255];
        }
        unsigned digit=0;while(digit<255 && rank>=bins[digit]) {rank-=bins[digit];++digit;}
        prefix|=static_cast<unsigned long long>(digit)<<(8*byte);prefix_mask|=255ull<<(8*byte);
    }
    return key_value<T>(prefix);
}
template<class T> __global__ void selection(std::uint64_t n, const std::uint64_t* genes,
    DenseView gene_mask, Result r, Options o, bool stochastic) {
    if(blockIdx.x || threadIdx.x) return;
    std::uint64_t selected=0,eligible=0;
    for(std::uint64_t g=0;g<n;++g) {selected+=bool(r.selected[g]);eligible+=r.initial_status[g]==0 && enabled(gene_mask,genes[g]);}
    *r.low_signal=0;*r.refit_all=0;
    if(o.low_signal && selected<2) {
        *r.low_signal=1;
        if(eligible) {
            T q=T(.8)*T(eligible-1);auto lo=std::uint64_t(q),hi=lo+1<eligible?lo+1:lo;
            T a=eligible_order_statistic<T>(n,genes,gene_mask,r,lo),b=eligible_order_statistic<T>(n,genes,gene_mask,r,hi);
            T threshold=a+(q-T(lo))*(b-a);selected=0;
            for(std::uint64_t g=0;g<n;++g) {r.selected[g]=r.initial_status[g]==0 && enabled(gene_mask,genes[g]) && static_cast<T*>(r.r2)[g]>threshold;selected+=bool(r.selected[g]);}
        }
    }
    if(stochastic) {
        *r.refit_all=o.refit_all_if_none && selected==0;
        for(std::uint64_t g=0;g<n;++g) {
            static_cast<T*>(r.gamma)[g]=static_cast<T*>(r.initial_gamma)[g];static_cast<T*>(r.offset)[g]=static_cast<T*>(r.initial_offset)[g];
            static_cast<T*>(r.offset2)[g]=0;static_cast<T*>(r.scale1)[g]=0;static_cast<T*>(r.scale2)[g]=0;static_cast<T*>(r.objective)[g]=0;
            r.status[g]=!enabled(gene_mask,genes[g]) || r.initial_status[g]==0
                ? std::uint8_t(sci::StochasticStatus::not_selected) : std::uint8_t(sci::StochasticStatus::invalid_initialization);
        }
    }
}
template<class T> __global__ void residual_scales(std::uint64_t rows, std::uint64_t first,
    std::uint64_t count, const std::uint64_t* genes, DenseView gene_mask, Fields f, Result r, Options o) {
    for(std::uint64_t j=blockIdx.x*blockDim.x+threadIdx.x;j<count;j+=std::uint64_t(gridDim.x)*blockDim.x) {
        auto g=first+j;if(!refine(r,g,genes[g],gene_mask))continue;
        auto p=j*rows;auto s=sci::scales(static_cast<T*>(f.ms)+p,static_cast<T*>(f.mu)+p,
            static_cast<T*>(f.mss)+p,static_cast<T*>(f.mus)+p,rows,1,initial<T>(r,g),
            sci::StochasticOptions<T>{options<T>(o),bool(o.second_intercept)},f.mask);
        static_cast<T*>(r.scale1)[g]=s.first;static_cast<T*>(r.scale2)[g]=s.second;r.status[g]=std::uint8_t(s.status);
    }
}
template<class T> __global__ void joint_fitting(std::uint64_t rows, std::uint64_t first,
    std::uint64_t count, const std::uint64_t* genes, DenseView gene_mask, Fields f, Result r, Options o) {
    for(std::uint64_t j=blockIdx.x*blockDim.x+threadIdx.x;j<count;j+=std::uint64_t(gridDim.x)*blockDim.x) {
        auto g=first+j;if(!refine(r,g,genes[g],gene_mask))continue;
        auto p=j*rows;auto fit=sci::joint_fit(static_cast<T*>(f.ms)+p,static_cast<T*>(f.mu)+p,
            static_cast<T*>(f.mss)+p,static_cast<T*>(f.mus)+p,rows,1,initial<T>(r,g),
            sci::Scales<T>{static_cast<T*>(r.scale1)[g],static_cast<T*>(r.scale2)[g],sci::StochasticStatus(r.status[g])},
            sci::StochasticOptions<T>{options<T>(o),bool(o.second_intercept)},static_cast<T*>(f.scores)+p,f.mask,sci::HeapSort{});
        static_cast<T*>(r.gamma)[g]=fit.gamma;static_cast<T*>(r.offset)[g]=fit.offset;static_cast<T*>(r.offset2)[g]=fit.offset2;
        static_cast<T*>(r.scale1)[g]=fit.scales.first;static_cast<T*>(r.scale2)[g]=fit.scales.second;
        static_cast<T*>(r.objective)[g]=fit.objective;r.status[g]=std::uint8_t(fit.status);
    }
}
template<class T> __global__ void output_fields(std::uint64_t rows, std::uint64_t first,
    std::uint64_t count, Fields f, Result r, Outputs out, bool stochastic) {
    for(std::uint64_t p=blockIdx.x*blockDim.x+threadIdx.x;p<rows*count;p+=std::uint64_t(gridDim.x)*blockDim.x) {
        auto i=p%rows,j=p/rows,g=first+j;T x=static_cast<T*>(f.ms)[p],y=static_cast<T*>(f.mu)[p];
        T gamma=static_cast<T*>(r.gamma)[g],offset=static_cast<T*>(r.offset)[g];
        auto target=out.block_local?j:g;
        if(out.velocity.values.data)store(out.velocity,i,target,y-gamma*x-offset);
        if(out.spliced.values.data)store(out.spliced,i,target,x);
        if(out.unspliced.values.data)store(out.unspliced,i,target,y);
        if(out.spliced_second.values.data)store(out.spliced_second,i,target,static_cast<T*>(f.mss)[p]);
        if(out.cross.values.data)store(out.cross,i,target,static_cast<T*>(f.mus)[p]);
        if(out.unspliced_second.values.data)store(out.unspliced_second,i,target,static_cast<T*>(f.muu)[p]);
        if(out.second_residual.values.data) {
            T value=0;
            if(stochastic && r.status[g]==0) value=(T(2)*static_cast<T*>(f.mus)[p]+y-T(2)*x*y)
                -gamma*(T(2)*static_cast<T*>(f.mss)[p]-x-T(2)*x*x)+T(2)*offset*x-static_cast<T*>(r.offset2)[g];
            store(out.second_residual,i,target,value);
        }
    }
}
template<class T> __global__ void neighborhood_rows(PreparedMatrix m, DenseView out) {
    for(std::uint64_t i=blockIdx.x*blockDim.x+threadIdx.x;i<out.rows;i+=std::uint64_t(gridDim.x)*blockDim.x) {
        for(std::uint64_t j=0;j<out.columns;++j)store(out,i,j,0.);
        if(!m.matrix.sparse) {
            for(std::uint64_t j=0;j<out.columns;++j) if(dense<T>(m.matrix.dense,i,j)>0)store(out,i,j,1.);
        } else {
            auto p=m.offsets[i],end=m.offsets[i+1];
            while(p<end) {
                auto column=m.edges[p].index;T sum=0;
                do {sum+=load<T>(m.matrix.compressed.values.data,m.matrix.compressed.dtype,m.edges[p].position);++p;}
                while(p<end && m.edges[p].index==column);
                if(sum>0)store(out,i,column,1.);
            }
        }
        store(out,i,i,1.);std::uint64_t count=0;
        for(std::uint64_t j=0;j<out.columns;++j)count+=dense<double>(out,i,j)>0;
        T weight=T(1./double(count));
        for(std::uint64_t j=0;j<out.columns;++j)if(dense<double>(out,i,j)>0)store(out,i,j,weight);
    }
}
}

int prepare_scratch(std::uint64_t major, std::size_t* bytes) {
    return cub::DeviceScan::ExclusiveSum(nullptr,*bytes,static_cast<const std::uint64_t*>(nullptr),static_cast<std::uint64_t*>(nullptr),major+1);
}
int validate(Matrix m, bool, int* error, void* stream) {
    auto n=m.sparse?m.compressed.nnz:m.dense.rows*m.dense.columns;
    if(n)validate_values<<<blocks(n),threads,0,native(stream)>>>(m,error);
    return cudaGetLastError();
}
int prepare(Matrix m, bool rows, std::uint64_t* offsets, Edge* edges, std::uint64_t* counts,
    void* scratch, std::size_t scratch_bytes, void* stream) {
    auto major=rows?m.compressed.rows:m.compressed.columns;
    auto source_major=m.compressed.layout==SparseLayout::csr?m.compressed.rows:m.compressed.columns;
    auto s=native(stream);auto e=cudaMemsetAsync(counts,0,(major+1)*sizeof(std::uint64_t),s);if(e)return e;
    if(source_major)count_major<<<blocks(source_major),threads,0,s>>>(m.compressed,rows,counts);
    e=cudaGetLastError();if(e)return e;
    e=cub::DeviceScan::ExclusiveSum(scratch,scratch_bytes,counts,offsets,major+1,s);if(e)return e;
    e=cudaMemsetAsync(counts,0,(major+1)*sizeof(std::uint64_t),s);if(e)return e;
    if(source_major)fill_edges<<<blocks(source_major),threads,0,s>>>(m.compressed,rows,offsets,counts,edges);
    if(major)sort_edges<<<blocks(major),threads,0,s>>>(major,offsets,edges);
    return cudaGetLastError();
}
int pack(PreparedMatrix s, PreparedMatrix u, const std::uint64_t* genes,
    std::uint64_t first,std::uint64_t count,std::uint64_t rows,Fields f,bool fp32,void* stream) {
    int e=fp32?pack_one<float>(s,genes,first,nullptr,count,rows,f.source_s,native(stream))
              :pack_one<double>(s,genes,first,nullptr,count,rows,f.source_s,native(stream));if(e)return e;
    return fp32?pack_one<float>(u,genes,first,nullptr,count,rows,f.source_u,native(stream))
               :pack_one<double>(u,genes,first,nullptr,count,rows,f.source_u,native(stream));
}
int pack_override(PreparedMatrix s,PreparedMatrix u,std::uint64_t gene,std::uint64_t destination,
    std::uint64_t rows,Fields f,bool fp32,void* stream) {
    auto offset=destination*rows*(fp32?sizeof(float):sizeof(double));
    auto* x=static_cast<char*>(f.source_s)+offset;auto* y=static_cast<char*>(f.source_u)+offset;
    int e=fp32?pack_one<float>(s,nullptr,gene,nullptr,1,rows,x,native(stream))
              :pack_one<double>(s,nullptr,gene,nullptr,1,rows,x,native(stream));if(e)return e;
    return fp32?pack_one<float>(u,nullptr,gene,nullptr,1,rows,y,native(stream))
               :pack_one<double>(u,nullptr,gene,nullptr,1,rows,y,native(stream));
}
int moments(PreparedMatrix graph,std::uint64_t rows,std::uint64_t count,Fields f,bool fp32,bool second,void* stream) {
    if(rows && count) {
        if(fp32)moment_fields<float><<<blocks(rows*count),threads,0,native(stream)>>>(graph,rows,count,f,second);
        else moment_fields<double><<<blocks(rows*count),threads,0,native(stream)>>>(graph,rows,count,f,second);
    }
    return cudaGetLastError();
}
int initial_fit(std::uint64_t rows,std::uint64_t first,std::uint64_t count,const std::uint64_t* genes,
    DenseView mask,DenseView gene_mask,Fields f,Result r,Options o,bool fp32,void* stream) {
    if(mask.values.data && rows)pack_mask<<<blocks(rows),threads,0,native(stream)>>>(mask,rows,f.mask);
    if(count) {
        if(fp32)initialize<float><<<blocks(count),threads,0,native(stream)>>>(rows,first,count,genes,gene_mask,f,r,o);
        else initialize<double><<<blocks(count),threads,0,native(stream)>>>(rows,first,count,genes,gene_mask,f,r,o);
    }
    return cudaGetLastError();
}
int resolve(std::uint64_t genes,const std::uint64_t* source_genes,DenseView gene_mask,Result r,Options o,void*,bool fp32,bool stochastic,void* stream) {
    if(fp32)selection<float><<<1,1,0,native(stream)>>>(genes,source_genes,gene_mask,r,o,stochastic);
    else selection<double><<<1,1,0,native(stream)>>>(genes,source_genes,gene_mask,r,o,stochastic);
    return cudaGetLastError();
}
int scales(std::uint64_t rows,std::uint64_t first,std::uint64_t count,const std::uint64_t* genes,
    DenseView mask,DenseView gene_mask,Fields f,Result r,Options o,bool fp32,void* stream) {
    if(mask.values.data && rows)pack_mask<<<blocks(rows),threads,0,native(stream)>>>(mask,rows,f.mask);
    if(count) {
        if(fp32)residual_scales<float><<<blocks(count),threads,0,native(stream)>>>(rows,first,count,genes,gene_mask,f,r,o);
        else residual_scales<double><<<blocks(count),threads,0,native(stream)>>>(rows,first,count,genes,gene_mask,f,r,o);
    }
    return cudaGetLastError();
}
int fit(std::uint64_t rows,std::uint64_t first,std::uint64_t count,const std::uint64_t* genes,
    DenseView,DenseView gene_mask,Fields f,Result r,Options o,bool fp32,void* stream) {
    if(count) {
        if(fp32)joint_fitting<float><<<blocks(count),threads,0,native(stream)>>>(rows,first,count,genes,gene_mask,f,r,o);
        else joint_fitting<double><<<blocks(count),threads,0,native(stream)>>>(rows,first,count,genes,gene_mask,f,r,o);
    }
    return cudaGetLastError();
}
int emit(std::uint64_t rows,std::uint64_t first,std::uint64_t count,Fields f,Result r,Outputs out,bool fp32,bool stochastic,void* stream) {
    if(rows && count) {
        if(fp32)output_fields<float><<<blocks(rows*count),threads,0,native(stream)>>>(rows,first,count,f,r,out,stochastic);
        else output_fields<double><<<blocks(rows*count),threads,0,native(stream)>>>(rows,first,count,f,r,out,stochastic);
    }
    return cudaGetLastError();
}
int neighborhood(PreparedMatrix m,DenseView out,void* stream) {
    if(out.rows) {
        if(out.dtype==ScalarType::float64)neighborhood_rows<double><<<blocks(out.rows),threads,0,native(stream)>>>(m,out);
        else neighborhood_rows<float><<<blocks(out.rows),threads,0,native(stream)>>>(m,out);
    }
    return cudaGetLastError();
}
}
