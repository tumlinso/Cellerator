module;
#include <Cellerator/ops/numeric/binary16.hh>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>
#include <type_traits>

module velocity;
import anndatapp.core;
#include "detail/prepared.hh"
#include "detail/stochastic_fit.hh"

namespace velocity {
namespace detail {

std::size_t add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a) throw std::invalid_argument("size overflow");
    return a + b;
}
std::size_t mul(std::size_t a, std::size_t b) {
    if (a && b > std::numeric_limits<std::size_t>::max()/a) throw std::invalid_argument("size overflow");
    return a*b;
}

template<class O,class T> O output_cast(T value) {
    if constexpr(std::is_same_v<O,std::uint16_t>) return cellerator::numeric::round_f16(value);
    else return O(value);
}
template<class F> void output_dispatch(StorageType type,F&& f) {
    if(type==StorageType::float16) f.template operator()<std::uint16_t>();
    else if(type==StorageType::float32) f.template operator()<float>();
    else f.template operator()<double>();
}
struct Mask {
    std::optional<anndatapp::ArrayView> view;
    const std::byte* data = nullptr;
    std::ptrdiff_t stride = 0;
    anndatapp::DType dtype = anndatapp::DType::uint8;
    Mask() = default;
    Mask(std::optional<anndatapp::ArrayView> v, std::size_t size) : view(std::move(v)) {
        if (!view) return;
        view->validate();
        if (view->rank() != 1 || view->extent(0) != size || view->layout() != anndatapp::Layout::dense ||
            (view->dtype() != anndatapp::DType::boolean && view->dtype() != anndatapp::DType::uint8))
            throw std::invalid_argument("mask must be a dense boolean or uint8 vector of the matching length");
        auto b = view->bytes(); data = b.data(); stride = b.stride_bytes(0); dtype = view->dtype();
    }
    bool get(std::size_t i) const {
        if (!view) return true;
        auto p = data + std::ptrdiff_t(i)*stride;
        if (dtype == anndatapp::DType::boolean) { bool x; std::memcpy(&x, p, 1); return x; }
        return std::to_integer<unsigned char>(*p) != 0;
    }
};

struct PreparedState {
    PreparedInputs input;
    NumericView weights, spliced, unspliced;
    std::vector<std::size_t> genes;
    Mask fit_mask, gene_mask;
    StorageType output_type;
    bool checked;
    PreparedState(Input in, anndatapp::Validation validation) : input(in.weights, in.spliced, in.unspliced),
        weights(in.weights), spliced(in.spliced), unspliced(in.unspliced),
        fit_mask(std::move(in.fit_mask), input.n_obs()), gene_mask(std::move(in.gene_mask), input.n_vars()),
        output_type(in.spliced.dtype() == StorageType::float64 || in.unspliced.dtype() == StorageType::float64
                    ? StorageType::float64 : StorageType::float32), checked(validation == anndatapp::Validation::full) {
        // Selection validates Axis arithmetic and bounds before unchecked index().
        auto axis = anndatapp::detail::compose_axis(anndatapp::Axis::all(), in.genes, input.n_vars());
        auto n = axis.extent(input.n_vars());
        genes.resize(n);
        for (std::size_t i = 0; i < n; ++i) genes[i] = axis.index(i);
        if (checked) input.check_values();
    }
};

template<class T> void storage(std::vector<T>& v, std::size_t size) {
    if (v.size() != size) { std::vector<T>().swap(v); v.resize(size); }
}
template<class T> struct BasicWorkspaceState {
    const PreparedState* owner;
    RunOptions options;
    StorageType output_type;
    std::size_t n, genes, block = 0, scratch = 0, width, slots = 0;
    std::vector<T> ms, mu, score, selection_score;
    std::vector<double> out64;
    std::vector<std::uint16_t> out16;
    std::vector<float> out32;
    std::vector<std::uint8_t> mask;
    BasicWorkspaceState(const PreparedInput&, const PreparedState& p, RunOptions o)
        : owner(&p), options(o), output_type(o.output_dtype.value_or(std::is_same_v<T,float> ? StorageType::float32 : p.output_type)),
          n(p.input.n_obs()), genes(p.genes.size()), width(output_type == StorageType::float16 ? 2 : output_type == StorageType::float32 ? 4 : 8) {
        if (!o.block_genes || (output_type != StorageType::float16 && output_type != StorageType::float32 && output_type != StorageType::float64))
            throw std::invalid_argument("invalid workspace options");
        configure(false, false);
    }
    void configure(bool selection, bool sink) {
        slots = sink ? std::size_t(options.velocity) + (output_type==(std::is_same_v<T,float> ? StorageType::float32 : StorageType::float64) ? 0 : std::size_t(options.spliced_moments)+options.unspliced_moments) : 0;
        auto fixed = add(mul(n,sizeof(T)+1), selection ? mul(genes,sizeof(T)) : 0);
        auto per = mul(n,add(2*sizeof(T),mul(slots,width)));
        if (fixed > options.workspace_bytes || (genes && per > options.workspace_bytes-fixed))
            throw std::invalid_argument("workspace budget cannot hold one gene");
        auto limit = std::min<std::size_t>({options.block_genes,64,std::max<std::size_t>(genes,1)});
        auto next = per ? std::min(limit,(options.workspace_bytes-fixed)/per) : limit;
        if (!next) throw std::invalid_argument("workspace budget cannot hold one gene");
        block = block ? std::min(block,next) : next;
        auto count = mul(n,block);
        // Drop shrinking sink storage before any new selection or worker allocation.
        if (out16.size()>(width==2 ? mul(count,slots) : 0)) std::vector<std::uint16_t>().swap(out16);
        if (out32.size()>(width==4 ? mul(count,slots) : 0)) std::vector<float>().swap(out32);
        if (out64.size()>(width==8 ? mul(count,slots) : 0)) std::vector<double>().swap(out64);
        storage(ms,count); storage(mu,count); storage(score,n); storage(mask,n);
        storage(selection_score,selection ? genes : 0);
        refresh_masks();
        storage(out16,width==2 ? mul(count,slots) : 0);
        storage(out32,width==4 ? mul(count,slots) : 0);
        storage(out64,width==8 ? mul(count,slots) : 0);
        scratch = add(mask.capacity(),add(mul(out16.capacity(),2),add(mul(out32.capacity(),4),mul(out64.capacity(),8))));
        for (auto* v : {&ms,&mu,&score,&selection_score}) scratch=add(scratch,mul(v->capacity(),sizeof(T)));
        if (scratch > options.workspace_bytes) throw std::invalid_argument("allocated workspace exceeds budget");
    }
    void refresh_masks() { for(std::size_t i=0;i<n;++i)mask[i]=std::uint8_t(owner->fit_mask.get(i)); }
    NumericBlock view(std::size_t slot, std::size_t count) const {
        const void* data = width==2 ? static_cast<const void*>(out16.data()) : width==4 ? static_cast<const void*>(out32.data()) : static_cast<const void*>(out64.data());
        auto* p=static_cast<const std::byte*>(data);
        if(n) p+=slot*n*block*width;
        return {output_type,{p,n,count,std::ptrdiff_t(count*width),std::ptrdiff_t(width),width,true}};
    }
};

template<class T> struct BasicStagesState {
    const PreparedState* p;
    BasicWorkspaceState<T>* w;
    FitOptions options;
    BlockSink sink;
    FitResult result;
    T *gamma, *offset, *r2;
    bool* selected;
    std::uint8_t* status;
    std::size_t next = 0;
    bool finished = false;
    FitBuffers<T> buffers() const { return {p->genes.size(),gamma,offset,r2,selected,status}; }
    BasicStagesState(const PreparedState& ps, BasicWorkspaceState<T>& ws, FitOptions o, BlockSink b)
        : p(&ps), w(&ws), options(o), sink(b),
          result{anndatapp::Array::empty<T>({ps.genes.size()}), anndatapp::Array::empty<T>({ps.genes.size()}),
                 anndatapp::Array::empty<T>({ps.genes.size()}), anndatapp::Array::empty<bool>({ps.genes.size()}),
                 anndatapp::Array::empty<std::uint8_t>({ps.genes.size()})} {
        gamma = result.gamma.edit().dense<T>().data(); offset = result.offset.edit().dense<T>().data();
        r2 = result.r2.edit().dense<T>().data(); selected = result.genes.edit().dense<bool>().data();
        status = result.status.edit().dense<std::uint8_t>().data();
        T validation_scratch=0;
        reference::fit_gene(static_cast<const T*>(nullptr),static_cast<const T*>(nullptr),0,1,o,&validation_scratch);
        if (!ps.genes.empty()) {
            std::fill_n(gamma, ps.genes.size(), 0); std::fill_n(offset, ps.genes.size(), 0);
            std::fill_n(r2, ps.genes.size(), 0); std::fill_n(selected, ps.genes.size(), false);
            std::fill_n(status, ps.genes.size(), std::uint8_t(FitStatus::insufficient_data));
        }
        w->configure(o.low_signal == LowSignal::scvelo_percentile, b.write != nullptr);
        if (p->checked) p->input.check_values();
        for (std::size_t i = 0; i < w->n; ++i) w->mask[i] = std::uint8_t(p->fit_mask.get(i));
    }
};

template<class T> struct BasicStochasticWorkspaceState {
    const PreparedState* owner;
    StochasticRunOptions options;
    StorageType dtype;
    std::size_t n, block = 0, bytes = 0, slots = 0, elements = 0;
    std::vector<T> ms,mu,mss,mus,muu,source_s,source_u,score,selection_score;
    std::vector<double> out64;
    std::vector<std::uint16_t> out16;
    std::vector<float> out32;
    std::vector<std::uint8_t> mask;
    BasicStochasticWorkspaceState(const PreparedState& p, StochasticRunOptions o)
        : owner(&p),options(o),dtype(o.output_dtype.value_or(std::is_same_v<T,float> ? StorageType::float32 : p.output_type)),n(p.input.n_obs()) {
        if (!o.block_genes || (dtype!=StorageType::float16 && dtype!=StorageType::float32 && dtype!=StorageType::float64))
            throw std::invalid_argument("invalid stochastic workspace options");
        configure(false,false);
    }
    void configure(bool selection,bool sink) {
        slots=sink ? std::size_t(options.velocity)+options.second_residual+(dtype==(std::is_same_v<T,float> ? StorageType::float32 : StorageType::float64) ? 0 : std::size_t(options.spliced_moments)+options.unspliced_moments+options.spliced_second+options.cross+options.unspliced_second) : 0;
        auto width=dtype==StorageType::float16 ? 2u : dtype==StorageType::float32 ? 4u : 8u;
        auto fixed=add(mul(n,3*sizeof(T)+1),selection ? mul(owner->genes.size(),sizeof(T)) : 0);
        auto per=mul(n,add((4+std::size_t(options.unspliced_second))*sizeof(T),mul(slots,width)));
        if(fixed>options.workspace_bytes || (!owner->genes.empty() && per>options.workspace_bytes-fixed))
            throw std::invalid_argument("stochastic workspace budget cannot hold one gene");
        auto limit=std::min<std::size_t>({options.block_genes,64,std::max<std::size_t>(owner->genes.size(),1)});
        auto next=per ? std::min(limit,(options.workspace_bytes-fixed)/per) : limit;
        if(!next)throw std::invalid_argument("stochastic workspace budget cannot hold one gene");
        block=block ? std::min(block,next) : next; elements=mul(n,block);
        // Drop shrinking sink storage before any new selection or worker allocation.
        if (out16.size()>(width==2 ? mul(elements,slots) : 0)) std::vector<std::uint16_t>().swap(out16);
        if (out32.size()>(width==4 ? mul(elements,slots) : 0)) std::vector<float>().swap(out32);
        if (out64.size()>(width==8 ? mul(elements,slots) : 0)) std::vector<double>().swap(out64);
        storage(ms,elements);storage(mu,elements);storage(mss,elements);storage(mus,elements);
        storage(muu,options.unspliced_second ? elements : 0);
        storage(source_s,n);storage(source_u,n);storage(score,n);storage(mask,n);
        storage(selection_score,selection ? owner->genes.size() : 0);
        refresh_masks();
        storage(out16,width==2 ? mul(elements,slots) : 0);storage(out32,width==4 ? mul(elements,slots) : 0);storage(out64,width==8 ? mul(elements,slots) : 0);
        bytes=add(mask.capacity(),add(mul(out16.capacity(),2),add(mul(out32.capacity(),4),mul(out64.capacity(),8))));
        for(auto* v:{&ms,&mu,&mss,&mus,&muu,&source_s,&source_u,&score,&selection_score})bytes=add(bytes,mul(v->capacity(),sizeof(T)));
        if(bytes>options.workspace_bytes)throw std::invalid_argument("allocated stochastic workspace exceeds budget");
    }
    void refresh_masks() { for(std::size_t i=0;i<n;++i)mask[i]=std::uint8_t(owner->fit_mask.get(i)); }
    NumericBlock view(std::size_t slot,std::size_t count) const {
        auto width=dtype==StorageType::float16 ? 2u : dtype==StorageType::float32 ? 4u : 8u;
        const void* data=width==2 ? static_cast<const void*>(out16.data()) : width==4 ? static_cast<const void*>(out32.data()) : static_cast<const void*>(out64.data());
        auto* p=static_cast<const std::byte*>(data);if(n)p+=slot*elements*width;
        return {dtype,{p,n,count,std::ptrdiff_t(count*width),std::ptrdiff_t(width),width,true}};
    }
};

template<class T> struct BasicStochasticStagesState {
    const PreparedState* p;
    BasicStochasticWorkspaceState<T>* w;
    StochasticOptions options;
    StochasticBlockSink sink;
    StochasticFitResult result;
    T *initial_gamma, *initial_offset, *r2, *gamma, *offset, *offset2, *scale1, *scale2, *objective;
    bool* selected;
    std::uint8_t *initial_status, *status;
    std::size_t next = 0;
    bool resolved = false, finished = false;
    BasicStochasticStagesState(const PreparedState& ps, BasicStochasticWorkspaceState<T>& ws, StochasticOptions o, StochasticBlockSink b)
        : p(&ps), w(&ws), options(o), sink(b) {
        auto g = p->genes.size();
        auto doubles = [g] { return anndatapp::Array::zeros<T>({g}); };
        result.initialization = {doubles(), doubles(), doubles(), anndatapp::Array::zeros<bool>({g}), anndatapp::Array::zeros<std::uint8_t>({g})};
        result.gamma = doubles(); result.offset = doubles(); result.offset2 = doubles(); result.first_scale = doubles();
        result.second_scale = doubles(); result.objective = doubles(); result.status = anndatapp::Array::zeros<std::uint8_t>({g});
        auto ptr = [](anndatapp::Array& a) { return a.edit().dense<T>().data(); };
        initial_gamma = ptr(result.initialization.gamma); initial_offset = ptr(result.initialization.offset); r2 = ptr(result.initialization.r2);
        gamma = ptr(result.gamma); offset = ptr(result.offset); offset2 = ptr(result.offset2);
        scale1 = ptr(result.first_scale); scale2 = ptr(result.second_scale); objective = ptr(result.objective);
        selected = result.initialization.genes.edit().dense<bool>().data();
        initial_status = result.initialization.status.edit().dense<std::uint8_t>().data();
        status = result.status.edit().dense<std::uint8_t>().data();
        std::fill_n(initial_status, g, std::uint8_t(FitStatus::insufficient_data));
        std::fill_n(status, g, std::uint8_t(StochasticStatus::not_selected));
        // Validate option contracts even for empty inputs, before any numerical work.
        T dummy = 0;
        reference::fit_gene(nullptr, nullptr, 0, 1, o.initialization, &dummy);
        w->configure(o.initialization.low_signal == LowSignal::scvelo_percentile, b.write != nullptr);
        if (p->checked) p->input.check_values();
        for (std::size_t i = 0; i < w->n; ++i) w->mask[i] = std::uint8_t(p->fit_mask.get(i));
    }
    StochasticFitBuffers<T> buffers() const { return {{p->genes.size(),initial_gamma,initial_offset,r2,selected,initial_status},gamma,offset,offset2,scale1,scale2,status,objective}; }

};
ComputePrecision resolve_precision(const PreparedState& p,ComputePrecision requested) {
    if(requested!=ComputePrecision::automatic)return requested;
    bool half=false,wide=false;
    for(auto type:{p.weights.dtype(),p.spliced.dtype(),p.unspliced.dtype()}) {
        half|=type==StorageType::float16;wide|=type==StorageType::float64;
    }
    return half&&!wide ? ComputePrecision::float32 : ComputePrecision::float64;
}
template<class T> constexpr ComputePrecision precision_of = std::is_same_v<T,float> ? ComputePrecision::float32 : ComputePrecision::float64;
struct WorkspaceState {
    const PreparedState* owner;
    ComputePrecision precision;
    std::unique_ptr<BasicWorkspaceState<float>> f32;
    std::unique_ptr<BasicWorkspaceState<double>> f64;
    WorkspaceState(const PreparedInput& p,const PreparedState& s,RunOptions o):owner(&s),precision(resolve_precision(s,o.compute_precision)) {
        if(precision==ComputePrecision::float32){f32=std::make_unique<BasicWorkspaceState<float>>(p,s,o);}
        else if(precision==ComputePrecision::float64){f64=std::make_unique<BasicWorkspaceState<double>>(p,s,o);}
        else throw std::invalid_argument("invalid compute precision");
    }
    template<class T> BasicWorkspaceState<T>& get() {
        if constexpr(std::is_same_v<T,float>){if(f32)return *f32;}else{if(f64)return *f64;}
        throw std::invalid_argument("stage precision does not match workspace");
    }
};
struct StochasticWorkspaceState {
    const PreparedState* owner;
    ComputePrecision precision;
    std::unique_ptr<BasicStochasticWorkspaceState<float>> f32;
    std::unique_ptr<BasicStochasticWorkspaceState<double>> f64;
    StochasticWorkspaceState(const PreparedState& p,StochasticRunOptions o):owner(&p),precision(resolve_precision(p,o.compute_precision)) {
        if(precision==ComputePrecision::float32){f32=std::make_unique<BasicStochasticWorkspaceState<float>>(p,o);}
        else if(precision==ComputePrecision::float64){f64=std::make_unique<BasicStochasticWorkspaceState<double>>(p,o);}
        else throw std::invalid_argument("invalid compute precision");
    }
    template<class T> BasicStochasticWorkspaceState<T>& get() {
        if constexpr(std::is_same_v<T,float>){if(f32)return *f32;}else{if(f64)return *f64;}
        throw std::invalid_argument("stage precision does not match workspace");
    }
};
struct Access {
    static const PreparedState* pointer(const PreparedInput& p){return p.state_.get();}
    static const WorkspaceState* pointer(const Workspace& w){return w.state_.get();}
    static const StochasticWorkspaceState* pointer(const StochasticWorkspace& w){return w.state_.get();}
    static const PreparedState& get(const PreparedInput& p){return *p.state_;}
    static WorkspaceState& get(Workspace& w){return *w.state_;}
    static StochasticWorkspaceState& get(StochasticWorkspace& w){return *w.state_;}
    template<class T> static BasicStagesState<T>& get(BasicReferenceStages<T>& s){return *s.state_;}
    template<class T> static BasicStochasticStagesState<T>& get(BasicStochasticReferenceStages<T>& s){return *s.state_;}
};
struct Span { std::uintptr_t begin = 0, end = 0; };
bool overlaps(Span a, Span b) { return a.begin < b.end && b.begin < a.end; }
std::size_t magnitude(std::ptrdiff_t x) {
    return x < 0 ? std::size_t(-(x + 1)) + 1 : std::size_t(x);
}
Span span(const std::byte* data, std::size_t rows, std::size_t cols,
          std::ptrdiff_t row_stride, std::ptrdiff_t col_stride, std::size_t width) {
    if (!rows || !cols) return {};
    std::size_t lo = 0, hi = 0;
    auto axis = [&](std::size_t n, std::ptrdiff_t stride) {
        auto d = mul(n - 1, magnitude(stride));
        if (stride < 0) lo = add(lo, d); else hi = add(hi, d);
    };
    axis(rows, row_stride); axis(cols, col_stride); hi = add(hi, width);
    auto base = reinterpret_cast<std::uintptr_t>(data);
    if (base < lo || hi > std::numeric_limits<std::uintptr_t>::max() - base)
        throw std::invalid_argument("buffer address range overflow");
    return {base - lo, base + hi};
}
Span contiguous_span(const std::byte* p, std::size_t n) {
    if (!n) return {};
    auto begin = reinterpret_cast<std::uintptr_t>(p);
    if (n > std::numeric_limits<std::uintptr_t>::max() - begin) throw std::invalid_argument("buffer address range overflow");
    return {begin, begin + n};
}
template<class View> bool aliases(View v, Span target) {
    if (v.layout() == anndatapp::Layout::dense) {
        auto b = v.bytes();
        auto n = v.extent(0), m = v.rank() == 1 ? 1 : v.extent(1);
        return overlaps(target, span(b.data(), n, m, b.stride_bytes(0), v.rank() == 1 ? 0 : b.stride_bytes(1), b.element_size()));
    }
    auto b = v.sparse_bytes();
    auto major = v.layout() == anndatapp::Layout::csr ? b.rows() : b.columns();
    return overlaps(target, contiguous_span(b.values(), mul(b.nnz(), b.value_size()))) ||
           overlaps(target, contiguous_span(b.indices(), mul(b.nnz(), b.index_size()))) ||
           overlaps(target, contiguous_span(b.offsets(), mul(add(major, 1), b.offset_size())));
}

void check_output_shape(NumericView v, std::size_t rows, std::size_t cols) {
    v.validate();
    if (v.layout() != anndatapp::Layout::dense || v.rank() != 2 || v.extent(0) != rows || v.extent(1) != cols ||
        (v.dtype() != StorageType::float16 && v.dtype() != StorageType::float32 && v.dtype() != StorageType::float64))
        throw std::invalid_argument("output must be a matching dense float16, float32 or float64 array");
    auto b = v.bytes(); auto rs = magnitude(b.stride_bytes(0)), cs = magnitude(b.stride_bytes(1));
    if ((rows > 1 && !rs) || (cols > 1 && !cs)) throw std::invalid_argument("output elements overlap");
    if (rows > 1 && cols > 1) {
        auto g = std::gcd(rs, cs);
        if (cs/g < rows && rs/g < cols) throw std::invalid_argument("output elements overlap");
    }
}

Span output_span(DenseOutput out) {
    auto width=out.type==StorageType::float16 ? 2u : out.type==StorageType::float32 ? 4u : 8u;
    return span(out.data,out.rows,out.columns,out.row_stride,out.column_stride,width);
}
template<class W> void check_workspace_alias(const W& w,Span target) {
    auto check=[&](const auto& v) {
        using Value=typename std::remove_cvref_t<decltype(v)>::value_type;
        if(overlaps(target,contiguous_span(reinterpret_cast<const std::byte*>(v.data()),mul(v.size(),sizeof(Value)))))throw std::invalid_argument("output aliases workspace");
    };
    check(w.ms);check(w.mu);check(w.score);check(w.selection_score);check(w.mask);
    check(w.out16);check(w.out32);check(w.out64);
    if constexpr(requires{w.mss;}) {check(w.mss);check(w.mus);check(w.muu);check(w.source_s);check(w.source_u);}
}

void check_input_alias(const PreparedState& p, Span target) {
    if (aliases(p.weights, target) || aliases(p.spliced, target) || aliases(p.unspliced, target) ||
        (p.fit_mask.view && aliases(*p.fit_mask.view, target)) || (p.gene_mask.view && aliases(*p.gene_mask.view, target)))
        throw std::invalid_argument("output aliases an input buffer");
}

void check_range(const PreparedState& p,GeneRange range,std::size_t capacity) {
    if(range.first>p.genes.size() || range.count>p.genes.size()-range.first || range.count>capacity)
        throw std::invalid_argument("gene block exceeds prepared input or workspace");
}
template<class T> void check_buffers(const PreparedState& p,FitBuffers<T> b) {
    if(b.size!=p.genes.size() || (b.size && (!b.gamma||!b.offset||!b.r2||!b.selected||!b.status)))
        throw std::invalid_argument("invalid fit diagnostic buffers");
}
template<class T> void check_buffers(const PreparedState& p,StochasticFitBuffers<T> b) {
    check_buffers(p,b.initialization);
    if(b.initialization.size && (!b.gamma||!b.offset||!b.offset2||!b.first_scale||!b.second_scale||!b.status||!b.objective))
        throw std::invalid_argument("invalid stochastic diagnostic buffers");
}
template<class T,class W> BasicFitBlock<T> fit_range(const PreparedState& p,W& w,BasicMomentBlock<T> m,FitBuffers<T> b,FitOptions options) {
    check_range(p,{m.first,m.count},w.block);check_buffers(p,b);
    if(m.rows!=w.n || (m.rows && m.count && (!m.spliced||!m.unspliced||!m.stride)))
        throw std::invalid_argument("invalid moment block");
    for(std::size_t j=0;j<m.count;++j) {
        auto k=m.first+j;
        auto f=reference::fit_gene(m.spliced?m.spliced+j:nullptr,m.unspliced?m.unspliced+j:nullptr,m.rows,m.stride,options,w.score.data(),w.mask.data(),p.gene_mask.get(p.genes[k]));
        b.gamma[k]=f.gamma;b.offset[k]=f.offset;b.r2[k]=f.r2;b.selected[k]=f.selected;b.status[k]=std::uint8_t(f.status);
    }
    return {m.first,m.count,b.gamma?b.gamma+m.first:nullptr,b.offset?b.offset+m.first:nullptr,b.r2?b.r2+m.first:nullptr,b.selected?b.selected+m.first:nullptr,b.status?b.status+m.first:nullptr};
}
template<class T> bool select_genes(const PreparedState& p,FitBuffers<T> b,FitOptions options,T* score,std::size_t capacity) {
    check_buffers(p,b);
    auto g=p.genes.size();
    for(std::size_t k=0;k<g;++k) {
        if(!std::isfinite(b.gamma[k])||!std::isfinite(b.offset[k])||!std::isfinite(b.r2[k])||b.status[k]>std::uint8_t(FitStatus::constant_response))throw std::invalid_argument("invalid fit diagnostics at global selection");
        b.selected[k]=b.selected[k] && b.status[k]==std::uint8_t(FitStatus::ok) && p.gene_mask.get(p.genes[k]);
    }
    if(options.low_signal!=LowSignal::scvelo_percentile || (g && std::count(b.selected,b.selected+g,true)>=2))return false;
    if(capacity<g || (g && !score))throw std::invalid_argument("global selection needs gene-sized score scratch");
    std::size_t n=0;
    for(std::size_t k=0;k<g;++k)
        if(b.status[k]==std::uint8_t(FitStatus::ok) && p.gene_mask.get(p.genes[k]))score[n++]=b.r2[k];
    if(n) {
        std::sort(score,score+n);
        T q=T(.8)*T(n-1);auto lo=std::size_t(q),hi=std::min(lo+1,n-1);
        T threshold=score[lo]+(q-T(lo))*(score[hi]-score[lo]);
        for(std::size_t k=0;k<g;++k)b.selected[k]=b.status[k]==std::uint8_t(FitStatus::ok) && p.gene_mask.get(p.genes[k]) && b.r2[k]>threshold;
    }
    return true;
}
template<class T> SelectionResult select_genes(const PreparedState& p,StochasticFitBuffers<T> b,StochasticOptions o,T* score,std::size_t capacity) {
    check_buffers(p,b);auto g=p.genes.size();
    bool low=select_genes(p,b.initialization,o.initialization,score,capacity);
    bool refit=o.refit_all_if_none && (!g || std::count(b.initialization.selected,b.initialization.selected+g,true)==0);
    for(std::size_t k=0;k<g;++k) {
        b.gamma[k]=b.initialization.gamma[k];b.offset[k]=b.initialization.offset[k];b.offset2[k]=0;
        b.first_scale[k]=b.second_scale[k]=b.objective[k]=0;
        b.status[k]=std::uint8_t(!p.gene_mask.get(p.genes[k]) || b.initialization.status[k]==std::uint8_t(FitStatus::ok) ? StochasticStatus::not_selected : StochasticStatus::invalid_initialization);
    }
    return {low,refit};
}
template<class T> BasicGeneFit<T> initial_fit(StochasticFitBuffers<T> b,std::size_t k) {
    auto a=b.initialization;return {a.gamma[k],a.offset[k],a.r2[k],FitStatus(a.status[k]),a.selected[k]};
}
template<class T> bool refine(const PreparedState& p,StochasticFitBuffers<T> b,std::size_t k,SelectionResult selection) {
    return (b.initialization.selected[k]||selection.refit_all) && p.gene_mask.get(p.genes[k]) && b.initialization.status[k]==std::uint8_t(FitStatus::ok);
}
template<class T,class W> BasicStochasticScaleBlock<T> scales_range(const PreparedState& p,W& w,BasicStochasticMomentBlock<T> m,StochasticFitBuffers<T> b,StochasticOptions o,SelectionResult selection) {
    check_range(p,{m.first,m.count},w.block);check_buffers(p,b);
    if(m.rows!=w.n || (m.rows&&m.count&&(!m.spliced||!m.unspliced||!m.spliced_second||!m.cross||!m.stride)))throw std::invalid_argument("invalid stochastic moment block");
    for(std::size_t j=0;j<m.count;++j) {
        auto k=m.first+j;if(!refine(p,b,k,selection))continue;
        auto r=reference::stochastic_scales(m.spliced?m.spliced+j:nullptr,m.unspliced?m.unspliced+j:nullptr,m.spliced_second?m.spliced_second+j:nullptr,m.cross?m.cross+j:nullptr,m.rows,m.stride,initial_fit(b,k),o,w.score.data(),w.mask.data());
        b.first_scale[k]=r.first;b.second_scale[k]=r.second;b.status[k]=std::uint8_t(r.status);
    }
    return {m.first,m.count,b.first_scale?b.first_scale+m.first:nullptr,b.second_scale?b.second_scale+m.first:nullptr,b.status?b.status+m.first:nullptr};
}
template<class T,class W> BasicStochasticFitBlock<T> fit_range(const PreparedState& p,W& w,BasicStochasticMomentBlock<T> m,BasicStochasticScaleBlock<T> scales,StochasticFitBuffers<T> b,StochasticOptions o,SelectionResult selection) {
    check_range(p,{m.first,m.count},w.block);check_buffers(p,b);
    if(scales.first!=m.first||scales.count!=m.count || (m.count&&(!scales.first_scale||!scales.second_scale||!scales.status)))throw std::invalid_argument("invalid stochastic scales");
    for(std::size_t j=0;j<m.count;++j) {
        auto k=m.first+j;if(!refine(p,b,k,selection))continue;
        if(scales.status[j]>std::uint8_t(StochasticStatus::nonfinite)||!std::isfinite(scales.first_scale[j])||!std::isfinite(scales.second_scale[j]))throw std::invalid_argument("invalid stochastic scales");
        auto f=reference::stochastic_fit_gene(m.spliced?m.spliced+j:nullptr,m.unspliced?m.unspliced+j:nullptr,m.spliced_second?m.spliced_second+j:nullptr,m.cross?m.cross+j:nullptr,m.rows,m.stride,initial_fit(b,k),{scales.first_scale[j],scales.second_scale[j],StochasticStatus(scales.status[j])},o,w.score.data(),w.mask.data());
        b.gamma[k]=f.gamma;b.offset[k]=f.offset;b.offset2[k]=f.offset2;b.first_scale[k]=f.scales.first;b.second_scale[k]=f.scales.second;b.status[k]=std::uint8_t(f.status);b.objective[k]=f.objective;
    }
    return {m.first,m.count,b.gamma?b.gamma+m.first:nullptr,b.offset?b.offset+m.first:nullptr,b.offset2?b.offset2+m.first:nullptr,b.first_scale?b.first_scale+m.first:nullptr,b.second_scale?b.second_scale+m.first:nullptr,b.status?b.status+m.first:nullptr,b.objective?b.objective+m.first:nullptr};
}

template<class T> void check_block(const BasicStagesState<T>& s, const BasicMomentBlock<T>& m) {
    if (s.finished || m.first != s.next || !m.count || m.count > s.w->block ||
        m.count > s.p->genes.size() - s.next || m.rows != s.w->n ||
        (m.rows && (!m.spliced || !m.unspliced || !m.stride)))
        throw std::invalid_argument("invalid or out-of-order moment block");
    if (m.rows) {
        auto limit = std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
        auto step = magnitude(m.stride);
        if (m.count - 1 > limit || (m.rows > 1 && step > (limit - (m.count - 1))/(m.rows - 1)))
            throw std::invalid_argument("moment block stride overflows");
    }
}

template<class T> void check_stochastic_range(const BasicStochasticStagesState<T>& s, std::size_t first, std::size_t count, bool resolved) {
    if (s.finished || s.resolved != resolved || first != s.next || !count || count > s.w->block ||
        first > s.p->genes.size() || count > s.p->genes.size() - first)
        throw std::invalid_argument("invalid or out-of-order stochastic block");
}
template<class T> void check_stochastic_block(const BasicStochasticStagesState<T>& s, const BasicStochasticMomentBlock<T>& m) {
    check_stochastic_range(s, m.first, m.count, true);
    if (m.rows != s.w->n || (m.rows && (!m.spliced || !m.unspliced || !m.spliced_second || !m.cross || !m.stride)))
        throw std::invalid_argument("invalid stochastic moment block");
    auto limit = std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if (m.count - 1 > limit || (m.rows > 1 && magnitude(m.stride) > (limit - (m.count - 1))/(m.rows - 1)))
        throw std::invalid_argument("stochastic stride overflows");
}
}

Input from_anndata(const anndatapp::AnnData& data, std::string_view s, std::string_view u, std::string_view w) {
    return {data.layer(s), data.layer(u), data.obsp(w), anndatapp::Axis::all(), std::nullopt, std::nullopt};
}

PreparedInput::PreparedInput(Input in, anndatapp::Validation validation) : state_(std::make_unique<detail::PreparedState>(std::move(in), validation)) {}
PreparedInput::~PreparedInput() = default;
PreparedInput::PreparedInput(PreparedInput&&) noexcept = default;
PreparedInput& PreparedInput::operator=(PreparedInput&&) noexcept = default;
std::size_t PreparedInput::n_cells() const noexcept { return state_->input.n_obs(); }
std::size_t PreparedInput::n_genes() const noexcept { return state_->genes.size(); }
std::size_t PreparedInput::source_genes() const noexcept { return state_->input.n_vars(); }
std::size_t PreparedInput::gene_index(std::size_t i) const {
    if (i >= n_genes()) throw std::out_of_range("gene index out of range");
    return state_->genes[i];
}
std::size_t PreparedInput::storage_bytes() const noexcept { return state_->input.prepared_bytes() + state_->genes.capacity()*sizeof(std::size_t); }
StorageType PreparedInput::output_dtype() const noexcept { return state_->output_type; }
void PreparedInput::validate_values() const { state_->input.check_values(); }

Workspace::Workspace(const PreparedInput& p, RunOptions o) : state_(std::make_unique<detail::WorkspaceState>(p, detail::Access::get(p), o)) {}
Workspace::~Workspace() = default;
Workspace::Workspace(Workspace&&) noexcept = default;
Workspace& Workspace::operator=(Workspace&&) noexcept = default;
std::size_t Workspace::block_genes() const noexcept { return state_->f32 ? state_->f32->block : state_->f64->block; }
std::size_t Workspace::scratch_bytes() const noexcept { return state_->f32 ? state_->f32->scratch : state_->f64->scratch; }

StochasticWorkspace::StochasticWorkspace(const PreparedInput& p, StochasticRunOptions o)
    : state_(std::make_unique<detail::StochasticWorkspaceState>(detail::Access::get(p), o)) {}
StochasticWorkspace::~StochasticWorkspace() = default;
StochasticWorkspace::StochasticWorkspace(StochasticWorkspace&&) noexcept = default;
StochasticWorkspace& StochasticWorkspace::operator=(StochasticWorkspace&&) noexcept = default;
std::size_t StochasticWorkspace::block_genes() const noexcept { return state_->f32 ? state_->f32->block : state_->f64->block; }
std::size_t StochasticWorkspace::scratch_bytes() const noexcept { return state_->f32 ? state_->f32->bytes : state_->f64->bytes; }

ComputePrecision Workspace::compute_precision() const noexcept {return state_->precision;}
ComputePrecision StochasticWorkspace::compute_precision() const noexcept {return state_->precision;}
NumericView PreparedInput::spliced() const {return state_->spliced;}
NumericView PreparedInput::unspliced() const {return state_->unspliced;}
NumericView PreparedInput::weights() const {return state_->weights;}
std::optional<anndatapp::ArrayView> PreparedInput::fit_mask() const {return state_->fit_mask.view;}
std::optional<anndatapp::ArrayView> PreparedInput::gene_mask() const {return state_->gene_mask.view;}
template<class T> BasicReferenceStages<T>::BasicReferenceStages(const PreparedInput& p, Workspace& w, FitOptions o, BlockSink b) {
    auto& ws = detail::Access::get(w).get<T>();
    if (ws.owner != &detail::Access::get(p)) throw std::invalid_argument("workspace belongs to another prepared input");
    if (b.write == nullptr && b.context != nullptr) throw std::invalid_argument("sink callback is null");
    state_ = std::make_unique<detail::BasicStagesState<T>>(detail::Access::get(p), ws, o, b);
}
template<class T> BasicReferenceStages<T>::~BasicReferenceStages() = default;
template<class T> BasicReferenceStages<T>::BasicReferenceStages(BasicReferenceStages<T>&&) noexcept = default;
template<class T> BasicReferenceStages<T>& BasicReferenceStages<T>::operator=(BasicReferenceStages<T>&&) noexcept = default;

template<class T> BasicMomentBlock<T> BasicReferenceStages<T>::moments(std::size_t first, std::size_t count) {
    auto& s = *state_; auto& w = *s.w;
    if (s.finished || first != s.next || !count || count > w.block || first > s.p->genes.size() || count > s.p->genes.size() - first)
        throw std::invalid_argument("invalid or out-of-order moment block");
    s.p->input.moments_selected(s.p->genes.data() + first, count, w.ms.data(), w.mu.data());
    return {first, count, w.n, std::ptrdiff_t(count), w.ms.data(), w.mu.data()};
}

template<class T> BasicFitBlock<T> BasicReferenceStages<T>::fit(BasicMomentBlock<T> m) {
    auto& s = *state_; auto& w = *s.w;
    detail::check_block(s, m);
    return detail::fit_range(*s.p,w,m,s.buffers(),s.options);
}

template<class T> void BasicReferenceStages<T>::emit(BasicMomentBlock<T> m, BasicFitBlock<T> fit) {
    auto& s = *state_; auto& w = *s.w;
    detail::check_block(s, m);
    if (fit.first != m.first || fit.count != m.count || !fit.gamma || !fit.offset || !fit.r2 || !fit.selected || !fit.status)
        throw std::invalid_argument("invalid fit block");
    for (std::size_t j = 0; j < m.count; ++j)
        if (!std::isfinite(fit.gamma[j]) || !std::isfinite(fit.offset[j]) || !std::isfinite(fit.r2[j]) ||
            fit.status[j] > std::uint8_t(FitStatus::constant_response))
            throw std::invalid_argument("fit block contains invalid diagnostics");
    for (std::size_t j = 0; j < m.count; ++j) {
        auto k = m.first + j;
        auto gamma = fit.gamma[j], offset = fit.offset[j], r2 = fit.r2[j];
        auto status = fit.status[j]; bool selected = fit.selected[j];
        s.gamma[k] = gamma; s.offset[k] = offset; s.r2[k] = r2; s.status[k] = status;
        s.selected[k] = selected && status == std::uint8_t(FitStatus::ok) && s.p->gene_mask.get(s.p->genes[k]);
    }
    if (s.sink.write) {
        OutputBlock out; out.first = m.first; out.count = m.count;
        std::size_t slot = 0;
        auto copy = [&]<bool Velocity>(const T* src) {
            if constexpr(!Velocity) if(w.output_type==(std::is_same_v<T,float> ? StorageType::float32 : StorageType::float64))
                return NumericBlock{w.output_type,{reinterpret_cast<const std::byte*>(src),m.rows,m.count,m.stride*std::ptrdiff_t(sizeof(T)),sizeof(T),sizeof(T),m.stride==std::ptrdiff_t(m.count)}};
            detail::output_dispatch(w.output_type,[&]<class O>() {
                O* dest;
                if constexpr(std::is_same_v<O,float>) dest=w.out32.data()+slot*w.n*w.block;
                else if constexpr(std::is_same_v<O,double>) dest=w.out64.data()+slot*w.n*w.block;
                else dest=w.out16.data()+slot*w.n*w.block;
                for(std::size_t i=0;i<m.rows;++i) for(std::size_t j=0;j<m.count;++j) {
                    auto q=std::ptrdiff_t(i)*m.stride+std::ptrdiff_t(j);
                    T value=Velocity ? m.unspliced[q]-s.gamma[m.first+j]*m.spliced[q]-s.offset[m.first+j] : src[q];
                    dest[i*m.count+j]=detail::output_cast<O>(value);
                }
            });
            return w.view(slot++,m.count);
        };
        if (w.options.velocity) out.velocity = copy.template operator()<true>(nullptr);
        if (w.options.spliced_moments) out.spliced = copy.template operator()<false>(m.spliced);
        if (w.options.unspliced_moments) out.unspliced = copy.template operator()<false>(m.unspliced);
        s.sink.write(s.sink.context, out);
    }
    s.next += m.count;
}

template<class T> FitResult BasicReferenceStages<T>::finish() {
    auto& s = *state_;
    if (s.finished || s.next != s.p->genes.size()) throw std::invalid_argument("reference stages have unfinished blocks");
    s.finished = true;
    s.result.low_signal=detail::select_genes(*s.p,s.buffers(),s.options,s.w->selection_score.data(),s.w->selection_score.size());
    return std::move(s.result);
}

template<class T> BasicStochasticReferenceStages<T>::BasicStochasticReferenceStages(const PreparedInput& p, StochasticWorkspace& w, StochasticOptions o, StochasticBlockSink b) {
    auto& ws = detail::Access::get(w).get<T>();
    if (ws.owner != &detail::Access::get(p)) throw std::invalid_argument("stochastic workspace belongs to another input");
    if (!b.write && b.context) throw std::invalid_argument("stochastic sink callback is null");
    state_ = std::make_unique<detail::BasicStochasticStagesState<T>>(detail::Access::get(p), ws, o, b);
}
template<class T> BasicStochasticReferenceStages<T>::~BasicStochasticReferenceStages() = default;
template<class T> BasicStochasticReferenceStages<T>::BasicStochasticReferenceStages(BasicStochasticReferenceStages<T>&&) noexcept = default;
template<class T> BasicStochasticReferenceStages<T>& BasicStochasticReferenceStages<T>::operator=(BasicStochasticReferenceStages<T>&&) noexcept = default;
template<class T> BasicMomentBlock<T> BasicStochasticReferenceStages<T>::initial_moments(std::size_t first, std::size_t count) {
    auto& s = *state_; auto& w = *s.w;
    detail::check_stochastic_range(s, first, count, false);
    s.p->input.moments_selected(s.p->genes.data() + first, count, w.ms.data(), w.mu.data());
    return {first, count, w.n, std::ptrdiff_t(count), w.ms.data(), w.mu.data()};
}
template<class T> BasicFitBlock<T> BasicStochasticReferenceStages<T>::initial_fit(BasicMomentBlock<T> m) {
    auto& s = *state_;
    detail::check_stochastic_range(s, m.first, m.count, false);
    if (m.rows != s.w->n || (m.rows && (!m.spliced || !m.unspliced || !m.stride))) throw std::invalid_argument("invalid initialization moments");
    auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if (m.count-1>limit || (m.rows>1 && detail::magnitude(m.stride)>(limit-(m.count-1))/(m.rows-1)))
        throw std::invalid_argument("initialization stride overflows");
    return detail::fit_range(*s.p,*s.w,m,s.buffers().initialization,s.options.initialization);
}
template<class T> void BasicStochasticReferenceStages<T>::accept_initialization(BasicMomentBlock<T> m, BasicFitBlock<T> f) {
    auto& s = *state_;
    detail::check_stochastic_range(s, m.first, m.count, false);
    if (f.first != m.first || f.count != m.count || !f.gamma || !f.offset || !f.r2 || !f.selected || !f.status)
        throw std::invalid_argument("invalid stochastic initialization fit");
    for (std::size_t j = 0; j < m.count; ++j) {
        if (!std::isfinite(f.gamma[j]) || !std::isfinite(f.offset[j]) || !std::isfinite(f.r2[j]) || f.status[j] > std::uint8_t(FitStatus::constant_response))
            throw std::invalid_argument("invalid stochastic initialization diagnostics");
        auto k = m.first+j;
        s.initial_gamma[k] = f.gamma[j]; s.initial_offset[k] = f.offset[j]; s.r2[k] = f.r2[j]; s.initial_status[k] = f.status[j];
        s.selected[k] = f.selected[j] && f.status[j] == std::uint8_t(FitStatus::ok) && s.p->gene_mask.get(s.p->genes[k]);
    }
    s.next += m.count;
}
template<class T> void BasicStochasticReferenceStages<T>::resolve_selection() {
    auto& s = *state_; auto g = s.p->genes.size();
    if (s.resolved || s.finished || s.next != g) throw std::invalid_argument("unfinished stochastic initialization");
    auto selection=detail::select_genes(*s.p,s.buffers(),s.options,s.w->selection_score.data(),s.w->selection_score.size());
    s.result.initialization.low_signal=selection.low_signal;s.result.refit_all=selection.refit_all;
    s.resolved = true; s.next = 0;
}
template<class T> BasicStochasticMomentBlock<T> BasicStochasticReferenceStages<T>::moments(std::size_t first, std::size_t count) {
    auto& s = *state_; auto& w = *s.w;
    detail::check_stochastic_range(s, first, count, true);
    // A standalone stage promises genuine moments even when custom fitting
    // chooses a different gene population from the built-in refinement policy.
    s.p->input.second_moments_selected(s.p->genes.data()+first,count,w.ms.data(),w.mu.data(),
        w.mss.data(),w.mus.data(),w.options.unspliced_second ? w.muu.data() : nullptr,
        w.source_s.data(),w.source_u.data());
    return {first,count,w.n,std::ptrdiff_t(count),w.ms.data(),w.mu.data(),w.mss.data(),w.mus.data(),
            w.options.unspliced_second ? w.muu.data() : nullptr};
}
template<class T> BasicStochasticScaleBlock<T> BasicStochasticReferenceStages<T>::scales(BasicStochasticMomentBlock<T> m) {
    auto& s = *state_; detail::check_stochastic_block(s,m);
    return detail::scales_range(*s.p,*s.w,m,s.buffers(),s.options,{s.result.initialization.low_signal,s.result.refit_all});
}
template<class T> BasicStochasticFitBlock<T> BasicStochasticReferenceStages<T>::fit(BasicStochasticMomentBlock<T> m,BasicStochasticScaleBlock<T> scales) {
    auto& s=*state_; detail::check_stochastic_block(s,m);
    return detail::fit_range(*s.p,*s.w,m,scales,s.buffers(),s.options,{s.result.initialization.low_signal,s.result.refit_all});
}
template<class T> void BasicStochasticReferenceStages<T>::emit(BasicStochasticMomentBlock<T> m,BasicStochasticFitBlock<T> f) {
    auto& s=*state_; auto& w=*s.w; detail::check_stochastic_block(s,m);
    if (f.first!=m.first || f.count!=m.count || !f.gamma || !f.offset || !f.offset2 || !f.first_scale || !f.second_scale || !f.status || !f.objective)
        throw std::invalid_argument("invalid stochastic fit block");
    for (std::size_t j=0;j<m.count;++j) {
        if (!std::isfinite(f.gamma[j]) || !std::isfinite(f.offset[j]) || !std::isfinite(f.offset2[j]) ||
            !std::isfinite(f.first_scale[j]) || !std::isfinite(f.second_scale[j]) || !std::isfinite(f.objective[j]) || f.status[j]>std::uint8_t(StochasticStatus::nonfinite))
            throw std::invalid_argument("invalid stochastic fit diagnostics");
        auto k=m.first+j;
        s.gamma[k]=f.gamma[j]; s.offset[k]=f.offset[j]; s.offset2[k]=f.offset2[j];
        s.scale1[k]=f.first_scale[j]; s.scale2[k]=f.second_scale[j]; s.status[k]=f.status[j]; s.objective[k]=f.objective[j];
    }
    if (s.sink.write) {
        StochasticOutputBlock out; out.first=m.first; out.count=m.count;
        std::size_t slot=0;
        auto copy=[&](int field) {
            if(field>=2 && w.dtype==(std::is_same_v<T,float> ? StorageType::float32 : StorageType::float64)) {
                const T* src=field==2?m.spliced:field==3?m.unspliced:field==4?m.spliced_second:field==5?m.cross:m.unspliced_second;
                if(!src && m.rows)throw std::invalid_argument("requested stochastic output is absent");
                return NumericBlock{w.dtype,{reinterpret_cast<const std::byte*>(src),m.rows,m.count,m.stride*std::ptrdiff_t(sizeof(T)),sizeof(T),sizeof(T),m.stride==std::ptrdiff_t(m.count)}};
            }
            detail::output_dispatch(w.dtype,[&]<class O>() {
                O* dest;
                if constexpr(std::is_same_v<O,float>) dest=w.out32.data()+slot*w.elements;
                else if constexpr(std::is_same_v<O,double>) dest=w.out64.data()+slot*w.elements;
                else dest=w.out16.data()+slot*w.elements;
                for(std::size_t i=0;i<m.rows;++i) for(std::size_t j=0;j<m.count;++j) {
                    auto q=std::ptrdiff_t(i)*m.stride+std::ptrdiff_t(j);auto k=m.first+j;
                    T x=m.spliced[q],y=m.unspliced[q],value=0;
                    if(field==0) value=y-s.gamma[k]*x-s.offset[k];
                    else if(field==1 && s.status[k]==std::uint8_t(StochasticStatus::ok))
                        value=(2*m.cross[q]+y-2*x*y)-s.gamma[k]*(2*m.spliced_second[q]-x-2*x*x)+2*s.offset[k]*x-s.offset2[k];
                    else if(field>=2) {
                        const T* source=field==2?m.spliced:field==3?m.unspliced:field==4?m.spliced_second:field==5?m.cross:m.unspliced_second;
                        if(!source && m.rows) throw std::invalid_argument("requested stochastic output is absent");
                        value=source[q];
                    }
                    dest[i*m.count+j]=detail::output_cast<O>(value);
                }
            });
            return w.view(slot++,m.count);
        };
        if(w.options.velocity) out.velocity=copy(0);
        if(w.options.second_residual) out.second_residual=copy(1);
        if(w.options.spliced_moments) out.spliced=copy(2);
        if(w.options.unspliced_moments) out.unspliced=copy(3);
        if(w.options.spliced_second) out.spliced_second=copy(4);
        if(w.options.cross) out.cross=copy(5);
        if(w.options.unspliced_second) out.unspliced_second=copy(6);
        s.sink.write(s.sink.context,out);
    }
    s.next+=m.count;
}
template<class T> StochasticFitResult BasicStochasticReferenceStages<T>::finish() {
    auto& s=*state_;
    if(s.finished || !s.resolved || s.next!=s.p->genes.size()) throw std::invalid_argument("unfinished stochastic blocks");
    s.finished=true; return std::move(s.result);
}

namespace reference {

template<class T> void moments_impl(const PreparedInput& p, std::size_t first, std::size_t count, T* ms, T* mu) {
    const auto& s = detail::Access::get(p);
    if (first > s.genes.size() || count > s.genes.size() - first || (count && s.input.n_obs() && (!ms || !mu)))
        throw std::invalid_argument("invalid moment output or range");
    (void)detail::mul(s.input.n_obs(), count);
    if (s.checked) s.input.check_values();
    if (!count) return;
    auto bytes = detail::mul(detail::mul(s.input.n_obs(), count), sizeof(T));
    auto a = detail::contiguous_span(reinterpret_cast<const std::byte*>(ms), bytes);
    auto b = detail::contiguous_span(reinterpret_cast<const std::byte*>(mu), bytes);
    if (detail::overlaps(a, b)) throw std::invalid_argument("moment outputs overlap");
    detail::check_input_alias(s, a); detail::check_input_alias(s, b);
    s.input.moments_selected(s.genes.data() + first, count, ms, mu);
}

template<class T> FitResult steady_state_impl(const PreparedInput& p, Workspace& w, FitOptions options, BlockSink sink) {
    BasicReferenceStages<T> stages(p, w, options, sink);
    return steady_state_with(stages, p.n_genes(), w.block_genes());
}

template<class T> void residuals_impl(const BasicMomentBlock<T>& m, const BasicFitBlock<T>& fit, NumericEdit output) {
    if (m.count != fit.count || m.first != fit.first || m.stride < std::ptrdiff_t(m.count) ||
        (m.rows && m.count && (!m.spliced || !m.unspliced || !fit.gamma || !fit.offset)))
        throw std::invalid_argument("incompatible moment and fit blocks");
    detail::check_output_shape(output.view(), m.rows, m.count);
    auto dst = output.bytes();
    auto target = detail::span(dst.data(), m.rows, m.count, dst.stride_bytes(0), dst.stride_bytes(1), dst.element_size());
    auto mr = detail::mul(detail::mul(std::size_t(m.stride), sizeof(T)), m.rows ? m.rows - 1 : 0);
    auto input_bytes = detail::add(mr, detail::mul(m.count, sizeof(T)));
    if (m.rows && m.count && (detail::overlaps(target, detail::contiguous_span(reinterpret_cast<const std::byte*>(m.spliced), input_bytes)) ||
                             detail::overlaps(target, detail::contiguous_span(reinterpret_cast<const std::byte*>(m.unspliced), input_bytes)) ||
                             detail::overlaps(target, detail::contiguous_span(reinterpret_cast<const std::byte*>(fit.gamma), detail::mul(m.count, sizeof(T)))) ||
                             detail::overlaps(target, detail::contiguous_span(reinterpret_cast<const std::byte*>(fit.offset), detail::mul(m.count, sizeof(T))))))
        throw std::invalid_argument("output aliases moment or fit buffers");
    auto rs = dst.stride_bytes(0), cs = dst.stride_bytes(1);
    auto write = [&]<class O>() {
        for (std::size_t i = 0; i < m.rows; ++i) for (std::size_t j = 0; j < m.count; ++j) {
            O value = detail::output_cast<O>(m.unspliced[i*std::size_t(m.stride) + j] - fit.gamma[j]*m.spliced[i*std::size_t(m.stride) + j] - fit.offset[j]);
            auto* ptr = dst.data() + std::ptrdiff_t(i)*rs + std::ptrdiff_t(j)*cs;
            std::memcpy(ptr, &value, sizeof value);
        }
    };
    detail::output_dispatch(output.view().dtype(),write);
}

template<class T> FitResult steady_state_into_impl(const PreparedInput& p, Workspace& w, NumericEdit output, FitOptions options) {
    auto destination=acquire_output(p,output);
    auto& ws=detail::Access::get(w).get<T>();
    detail::check_workspace_alias(ws,detail::output_span(destination));
    BasicReferenceStages<T> stages(p,w,options);
    for(std::size_t first=0;first<p.n_genes();) {
        auto count=std::min(w.block_genes(),p.n_genes()-first);
        auto m=stages.moments(first,count);auto f=stages.fit(m);
        reference::write_residuals(m,f,destination);stages.emit(m,f);first+=count;
    }
    return stages.finish();
}

}
namespace reference {
template<class T> void second_moments_impl(const PreparedInput& p,std::size_t first,std::size_t count,T* ms,T* mu,
                    T* mss,T* mus,T* muu,T* source_s,T* source_u) {
    auto& s=detail::Access::get(p);
    if(first>s.genes.size() || count>s.genes.size()-first || (count && s.input.n_obs() && ((!source_s && (ms||mss||mus)) || (!source_u && (mu||mus||muu)))))
        throw std::invalid_argument("invalid second moment range or scratch");
    if (!count || !s.input.n_obs()) return;
    if (s.checked) s.input.check_values();
    auto bytes=detail::mul(detail::mul(s.input.n_obs(),count),sizeof(T));
    std::array<detail::Span,7> buffers{};
    std::size_t n=0;
    for (auto* out : {ms,mu,mss,mus,muu}) if (out) buffers[n++]=detail::contiguous_span(reinterpret_cast<const std::byte*>(out),bytes);
    auto source_bytes=detail::mul(s.input.n_obs(),sizeof(T));
    if (ms||mss||mus) buffers[n++]=detail::contiguous_span(reinterpret_cast<const std::byte*>(source_s),source_bytes);
    if (mu||mus||muu) buffers[n++]=detail::contiguous_span(reinterpret_cast<const std::byte*>(source_u),source_bytes);
    for (std::size_t i=0;i<n;++i) {
        detail::check_input_alias(s,buffers[i]);
        for (std::size_t j=0;j<i;++j) if (detail::overlaps(buffers[i],buffers[j]))
            throw std::invalid_argument("second moment outputs or scratch overlap");
    }
    s.input.second_moments_selected(s.genes.data()+first,count,ms,mu,mss,mus,muu,source_s,source_u);
}
namespace {
template<class T> void check_stochastic_gene_impl(const T* ms,const T* mu,const T* mss,const T* mus,
                           std::size_t rows,std::ptrdiff_t stride,const StochasticOptions& o) {
    T dummy=0;
    reference::fit_gene(nullptr,nullptr,0,1,o.initialization,&dummy);
    if (!rows) return;
    if (!ms || !mu || !mss || !mus || !stride) throw std::invalid_argument("invalid stochastic gene buffers");
    auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if(rows>1 && detail::magnitude(stride)>limit/(rows-1)) throw std::invalid_argument("stochastic gene stride overflows");
}
}
template<class T> BasicStochasticScales<T> stochastic_scales_impl(const T* ms,const T* mu,const T* mss,const T* mus,
    std::size_t rows,std::ptrdiff_t stride,const BasicGeneFit<T>& initial,const StochasticOptions& o,T*,const std::uint8_t* mask) {
    check_stochastic_gene_impl(ms,mu,mss,mus,rows,stride,o);
    return detail::stochastic_scales(ms,mu,mss,mus,rows,stride,initial,o,mask);
}
template<class T> BasicStochasticGeneFit<T> stochastic_fit_gene_impl(const T* ms,const T* mu,const T* mss,const T* mus,
    std::size_t rows,std::ptrdiff_t stride,const BasicGeneFit<T>& initial,BasicStochasticScales<T> scales,const StochasticOptions& o,T* scratch,const std::uint8_t* mask) {
    check_stochastic_gene_impl(ms,mu,mss,mus,rows,stride,o);
    if (rows && o.initialization.trimming!=Trimming::none && !scratch) throw std::invalid_argument("stochastic fitting scratch is null");
    return detail::stochastic_fit_gene(ms,mu,mss,mus,rows,stride,initial,scales,o,scratch,mask);
}
template<class T> void stochastic_residuals_impl(const BasicStochasticMomentBlock<T>& m,const BasicStochasticFitBlock<T>& f,NumericEdit velocity,
                          std::optional<NumericEdit> second) {
    detail::check_output_shape(velocity.view(),m.rows,m.count);
    if(second) detail::check_output_shape(second->view(),m.rows,m.count);
    if(f.first!=m.first || f.count!=m.count) throw std::invalid_argument("incompatible stochastic fit block");
    if (!m.rows || !m.count) return;
    if (!f.gamma || !f.offset || !f.offset2 || !f.status || !m.spliced || !m.unspliced ||
        (second && (!m.spliced_second || !m.cross)) || !m.stride)
        throw std::invalid_argument("invalid stochastic residual inputs");
    auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if (m.count-1>limit || detail::magnitude(m.stride)>limit/sizeof(T) ||
        (m.rows>1 && detail::magnitude(m.stride)>(limit-(m.count-1))/(m.rows-1)))
        throw std::invalid_argument("stochastic residual stride overflows");
    std::array<detail::Span,2> targets{};
    auto target=[&](NumericEdit dest) {
        auto b=dest.bytes();
        return detail::span(b.data(),m.rows,m.count,b.stride_bytes(0),b.stride_bytes(1),b.element_size());
    };
    targets[0]=target(velocity); if(second) targets[1]=target(*second);
    if(second && detail::overlaps(targets[0],targets[1])) throw std::invalid_argument("stochastic residual outputs overlap");
    for (auto dest : targets) {
        for (auto* source : {m.spliced,m.unspliced,second?m.spliced_second:nullptr,second?m.cross:nullptr})
            if (source && detail::overlaps(dest,detail::span(reinterpret_cast<const std::byte*>(source),m.rows,m.count,
                    m.stride*std::ptrdiff_t(sizeof(T)),sizeof(T),sizeof(T))))
                throw std::invalid_argument("stochastic residual output aliases moments");
        for (auto* source : {f.gamma,f.offset,f.offset2})
            if(detail::overlaps(dest,detail::contiguous_span(reinterpret_cast<const std::byte*>(source),detail::mul(m.count,sizeof(T)))))
                throw std::invalid_argument("stochastic residual output aliases fit");
        if(detail::overlaps(dest,detail::contiguous_span(reinterpret_cast<const std::byte*>(f.status),m.count)))
            throw std::invalid_argument("stochastic residual output aliases status");
    }
    auto write=[&](NumericEdit dest,bool second_field) {
        auto b=dest.bytes();
        detail::output_dispatch(dest.view().dtype(),[&]<class O>() {
        for(std::size_t i=0;i<m.rows;++i) for(std::size_t j=0;j<m.count;++j) {
            auto q=std::ptrdiff_t(i)*m.stride+std::ptrdiff_t(j);
            T x=m.spliced[q], y=m.unspliced[q];
            T value=second_field ? (f.status[j]==std::uint8_t(StochasticStatus::ok)
                ? (2*m.cross[q]+y-2*x*y)-f.gamma[j]*(2*m.spliced_second[q]-x-2*x*x)+2*f.offset[j]*x-f.offset2[j] : 0)
                : y-f.gamma[j]*x-f.offset[j];
            auto* ptr=b.data()+std::ptrdiff_t(i)*b.stride_bytes(0)+std::ptrdiff_t(j)*b.stride_bytes(1);
            O v=detail::output_cast<O>(value);std::memcpy(ptr,&v,sizeof v);
        }
        });
    };
    write(velocity,false); if(second) write(*second,true);
}
template<class T> StochasticFitResult stochastic_impl(const PreparedInput& p,StochasticWorkspace& w,StochasticOptions o,StochasticBlockSink b) {
    BasicStochasticReferenceStages<T> stages(p,w,o,b);
    return stochastic_with(stages,p.n_genes(),w.block_genes());
}
template<class T> StochasticFitResult stochastic_into_impl(const PreparedInput& p,StochasticWorkspace& w,NumericEdit output,
                                   StochasticOptions o,std::optional<NumericEdit> second) {
    auto& ps=detail::Access::get(p); auto& ws=detail::Access::get(w).get<T>();
    if(ws.owner!=&ps) throw std::invalid_argument("stochastic workspace belongs to another input");
    auto destination=acquire_output(p,output);
    std::optional<DenseOutput> second_destination;if(second)second_destination=acquire_output(p,*second);
    auto target=detail::output_span(destination);detail::check_workspace_alias(ws,target);
    if(second_destination) {
        auto other=detail::output_span(*second_destination);detail::check_workspace_alias(ws,other);
        if(detail::overlaps(target,other))throw std::invalid_argument("stochastic outputs overlap");
    }
    BasicStochasticReferenceStages<T> stages(p,w,o);
    for(std::size_t first=0;first<p.n_genes();) {
        auto count=std::min(w.block_genes(),p.n_genes()-first); auto m=stages.initial_moments(first,count);
        stages.accept_initialization(m,stages.initial_fit(m)); first+=count;
    }
    stages.resolve_selection();
    for(std::size_t first=0;first<p.n_genes();) {
        auto count=std::min(w.block_genes(),p.n_genes()-first); auto m=stages.moments(first,count);
        auto f=stages.fit(m,stages.scales(m));
        reference::write_residuals(m,f,destination,second_destination);
        stages.emit(m,f); first+=count;
    }
    return stages.finish();
}

}

}
namespace velocity::reference {
#define VELOCITY_PRIMITIVE_WRAPPERS(T) \
void moments(const PreparedInput& p,std::size_t first,std::size_t count,T* ms,T* mu){moments_impl(p,first,count,ms,mu);} \
void residuals(const BasicMomentBlock<T>& m,const BasicFitBlock<T>& f,NumericEdit o){residuals_impl(m,f,o);} \
void second_moments(const PreparedInput& p,std::size_t first,std::size_t count,T* ms,T* mu,T* mss,T* mus,T* muu,T* ss,T* su){second_moments_impl(p,first,count,ms,mu,mss,mus,muu,ss,su);} \
BasicStochasticScales<T> stochastic_scales(const T* ms,const T* mu,const T* mss,const T* mus,std::size_t rows,std::ptrdiff_t stride,const BasicGeneFit<T>& initial,const StochasticOptions& o,T* scratch,const std::uint8_t* mask){return stochastic_scales_impl(ms,mu,mss,mus,rows,stride,initial,o,scratch,mask);} \
BasicStochasticGeneFit<T> stochastic_fit_gene(const T* ms,const T* mu,const T* mss,const T* mus,std::size_t rows,std::ptrdiff_t stride,const BasicGeneFit<T>& initial,BasicStochasticScales<T> scales,const StochasticOptions& o,T* scratch,const std::uint8_t* mask){return stochastic_fit_gene_impl(ms,mu,mss,mus,rows,stride,initial,scales,o,scratch,mask);} \
void stochastic_residuals(const BasicStochasticMomentBlock<T>& m,const BasicStochasticFitBlock<T>& f,NumericEdit output,std::optional<NumericEdit> second){stochastic_residuals_impl(m,f,output,second);}
VELOCITY_PRIMITIVE_WRAPPERS(float)
VELOCITY_PRIMITIVE_WRAPPERS(double)
#undef VELOCITY_PRIMITIVE_WRAPPERS
FitResult steady_state(const PreparedInput& p,Workspace& w,FitOptions o,BlockSink b){
    if(w.compute_precision()==ComputePrecision::float32)return steady_state_impl<float>(p,w,o,b);
    return steady_state_impl<double>(p,w,o,b);
}
FitResult steady_state_into(const PreparedInput& p,Workspace& w,NumericEdit output,FitOptions o){
    if(w.compute_precision()==ComputePrecision::float32)return steady_state_into_impl<float>(p,w,output,o);
    return steady_state_into_impl<double>(p,w,output,o);
}
StochasticFitResult stochastic(const PreparedInput& p,StochasticWorkspace& w,StochasticOptions o,StochasticBlockSink b){
    if(w.compute_precision()==ComputePrecision::float32)return stochastic_impl<float>(p,w,o,b);
    return stochastic_impl<double>(p,w,o,b);
}
StochasticFitResult stochastic_into(const PreparedInput& p,StochasticWorkspace& w,NumericEdit output,StochasticOptions o,std::optional<NumericEdit> second){
    if(w.compute_precision()==ComputePrecision::float32)return stochastic_into_impl<float>(p,w,output,o,second);
    return stochastic_into_impl<double>(p,w,output,o,second);
}
}
namespace velocity {
namespace detail {
template<class T,class Block> void check_replacement(const Block& m,std::size_t offset,std::size_t count,NumericView s,NumericView u) {
    s.validate();u.validate();
    if(offset>m.count || count>m.count-offset || s.layout()!=anndatapp::Layout::dense || u.layout()!=anndatapp::Layout::dense || s.rank()!=2 || u.rank()!=2 || s.extent(0)!=m.rows || u.extent(0)!=m.rows || s.extent(1)!=count || u.extent(1)!=count || m.stride<std::ptrdiff_t(m.count))
        throw std::invalid_argument("invalid alternate source block or destination range");
    auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if(m.rows>1 && std::size_t(m.stride)>limit/sizeof(T))throw std::invalid_argument("replacement stride overflows byte offsets");
    auto byte_stride=m.rows>1 ? m.stride*std::ptrdiff_t(sizeof(T)) : 0;
    std::array<T*,5> destinations{m.spliced,m.unspliced,nullptr,nullptr,nullptr};
    if constexpr(requires {m.spliced_second;}) {
        destinations[2]=m.spliced_second;destinations[3]=m.cross;destinations[4]=m.unspliced_second;
    }
    for(auto* destination:destinations) if(destination) {
        auto target=span(reinterpret_cast<const std::byte*>(destination),m.rows,m.count,byte_stride,sizeof(T),sizeof(T));
        if(aliases(s,target)||aliases(u,target))throw std::invalid_argument("alternate source aliases moments");
    }
}
}
template<class T> void BasicReferenceStages<T>::replace_moments(BasicMomentBlock<T>& m,std::size_t offset,std::size_t count,NumericView a,NumericView b) {
    auto& s=*state_;detail::check_block(s,m);detail::check_replacement<T>(m,offset,count,a,b);
    if(!count || !m.rows)return;
    s.p->input.moments_from(a,b,m.spliced+offset,m.unspliced+offset,std::size_t(m.stride),s.p->checked);
}
template<class T> void BasicStochasticReferenceStages<T>::replace_moments(BasicMomentBlock<T>& m,std::size_t offset,std::size_t count,NumericView a,NumericView b) {
    auto& s=*state_;detail::check_stochastic_range(s,m.first,m.count,false);
    if(m.rows!=s.w->n || (m.rows && (!m.spliced||!m.unspliced)))throw std::invalid_argument("invalid initialization moments");
    detail::check_replacement<T>(m,offset,count,a,b);
    if(!count || !m.rows)return;
    s.p->input.moments_from(a,b,m.spliced+offset,m.unspliced+offset,std::size_t(m.stride),s.p->checked);
}
template<class T> void BasicStochasticReferenceStages<T>::replace_moments(BasicStochasticMomentBlock<T>& m,std::size_t offset,std::size_t count,NumericView a,NumericView b) {
    auto& s=*state_;detail::check_stochastic_block(s,m);detail::check_replacement<T>(m,offset,count,a,b);
    if(!count || !m.rows)return;
    for(auto* scratch:{s.w->source_s.data(),s.w->source_u.data()}) {
        auto target=detail::contiguous_span(reinterpret_cast<const std::byte*>(scratch),detail::mul(m.rows,sizeof(T)));
        if(detail::aliases(a,target)||detail::aliases(b,target))throw std::invalid_argument("alternate source aliases source scratch");
    }
    s.p->input.second_moments_from(a,b,m.spliced+offset,m.unspliced+offset,m.spliced_second+offset,m.cross+offset,
        m.unspliced_second?m.unspliced_second+offset:nullptr,s.w->source_s.data(),s.w->source_u.data(),std::size_t(m.stride),s.p->checked);
}
namespace reference {
template<class T> void moments_from(const PreparedInput& p,NumericView s,NumericView u,T* ms,T* mu,std::size_t stride) {
    auto& state=detail::Access::get(p);s.validate();u.validate();
    if(s.rank()!=2 || u.rank()!=2 || s.extent(0)!=p.n_cells() || u.extent(0)!=p.n_cells() || s.extent(1)!=u.extent(1) || stride<s.extent(1))throw std::invalid_argument("invalid alternate source shapes or stride");
    if(stride>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(T))throw std::invalid_argument("alternate destination stride overflows byte offsets");
    auto count=s.extent(1);
    if(!count || !p.n_cells())return;
    if(!ms||!mu)throw std::invalid_argument("alternate moments output is null");
    auto a=detail::span(reinterpret_cast<const std::byte*>(ms),p.n_cells(),count,detail::mul(stride,sizeof(T)),sizeof(T),sizeof(T));
    auto b=detail::span(reinterpret_cast<const std::byte*>(mu),p.n_cells(),count,detail::mul(stride,sizeof(T)),sizeof(T),sizeof(T));
    if(detail::overlaps(a,b)||detail::aliases(s,a)||detail::aliases(s,b)||detail::aliases(u,a)||detail::aliases(u,b))throw std::invalid_argument("alternate source or moment buffers overlap");
    detail::check_input_alias(state,a);detail::check_input_alias(state,b);
    state.input.moments_from(s,u,ms,mu,stride,state.checked);
}
template<class T> void second_moments_from(const PreparedInput& p,NumericView s,NumericView u,T* ms,T* mu,T* mss,T* mus,T* muu,T* source_s,T* source_u,std::size_t stride) {
    auto& state=detail::Access::get(p);s.validate();u.validate();
    if(s.rank()!=2 || u.rank()!=2 || s.extent(0)!=p.n_cells() || u.extent(0)!=p.n_cells() || s.extent(1)!=u.extent(1) || stride<s.extent(1))throw std::invalid_argument("invalid alternate source shapes or stride");
    if(stride>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(T))throw std::invalid_argument("alternate destination stride overflows byte offsets");
    auto count=s.extent(1);
    if(!count || !p.n_cells())return;
    if((ms||mss||mus)&&!source_s)throw std::invalid_argument("alternate source_s scratch is null");
    if((mu||mus||muu)&&!source_u)throw std::invalid_argument("alternate source_u scratch is null");
    std::array<detail::Span,7> spans{};std::size_t n=0;
    for(auto* out:{ms,mu,mss,mus,muu})if(out)spans[n++]=detail::span(reinterpret_cast<const std::byte*>(out),p.n_cells(),count,detail::mul(stride,sizeof(T)),sizeof(T),sizeof(T));
    if(ms||mss||mus)spans[n++]=detail::contiguous_span(reinterpret_cast<const std::byte*>(source_s),detail::mul(p.n_cells(),sizeof(T)));
    if(mu||mus||muu)spans[n++]=detail::contiguous_span(reinterpret_cast<const std::byte*>(source_u),detail::mul(p.n_cells(),sizeof(T)));
    for(std::size_t i=0;i<n;++i) {
        detail::check_input_alias(state,spans[i]);
        if(detail::aliases(s,spans[i])||detail::aliases(u,spans[i]))throw std::invalid_argument("alternate source aliases moment buffers");
        for(std::size_t j=0;j<i;++j)if(detail::overlaps(spans[i],spans[j]))throw std::invalid_argument("alternate moment buffers overlap");
    }
    state.input.second_moments_from(s,u,ms,mu,mss,mus,muu,source_s,source_u,stride,state.checked);
}
#define VELOCITY_ALTERNATE_INSTANTIATE(T) \
template void moments_from<T>(const PreparedInput&,NumericView,NumericView,T*,T*,std::size_t); \
template void second_moments_from<T>(const PreparedInput&,NumericView,NumericView,T*,T*,T*,T*,T*,T*,T*,std::size_t);
VELOCITY_ALTERNATE_INSTANTIATE(float)
VELOCITY_ALTERNATE_INSTANTIATE(double)
#undef VELOCITY_ALTERNATE_INSTANTIATE
}
}
namespace velocity {
template<class T> bool BasicReferenceStages<T>::bound_to(const PreparedInput& p,const Workspace& w) const noexcept {
    auto* ws=detail::Access::pointer(w);
    if(!state_||!ws||state_->p!=detail::Access::pointer(p))return false;
    if constexpr(std::is_same_v<T,float>)return state_->w==ws->f32.get();
    else return state_->w==ws->f64.get();
}
template<class T> bool BasicStochasticReferenceStages<T>::bound_to(const PreparedInput& p,const StochasticWorkspace& w) const noexcept {
    auto* ws=detail::Access::pointer(w);
    if(!state_||!ws||state_->p!=detail::Access::pointer(p))return false;
    if constexpr(std::is_same_v<T,float>)return state_->w==ws->f32.get();
    else return state_->w==ws->f64.get();
}
}
template class velocity::BasicReferenceStages<float>;
template class velocity::BasicReferenceStages<double>;
template class velocity::BasicStochasticReferenceStages<float>;
template class velocity::BasicStochasticReferenceStages<double>;

namespace velocity {
void validate_run(const PreparedInput& p) { const auto& s=detail::Access::get(p);if(s.checked)s.input.check_values(); }
void Workspace::refresh_masks() { auto& w=*state_;if(w.f32)w.f32->refresh_masks();else w.f64->refresh_masks(); }
void StochasticWorkspace::refresh_masks() { auto& w=*state_;if(w.f32)w.f32->refresh_masks();else w.f64->refresh_masks(); }
DenseOutput acquire_output(const PreparedInput& p,NumericEdit edit) {
    auto view=edit.view();view.validate();
    if(view.layout()!=anndatapp::Layout::dense||view.rank()!=2||view.extent(0)!=p.n_cells()||view.extent(1)!=p.n_genes())throw std::invalid_argument("output must be a matching dense numeric matrix");
    auto b=edit.bytes();auto rows=p.n_cells(),cols=p.n_genes();
    auto rs=detail::magnitude(b.stride_bytes(0)),cs=detail::magnitude(b.stride_bytes(1));
    if((rows>1&&!rs)||(cols>1&&!cs))throw std::invalid_argument("output elements overlap");
    if(rows>1&&cols>1) {auto g=std::gcd(rs,cs);if(cs/g<rows&&rs/g<cols)throw std::invalid_argument("output elements overlap");}
    DenseOutput out{b.data(),rows,cols,b.stride_bytes(0),b.stride_bytes(1),view.dtype()};
    detail::check_input_alias(detail::Access::get(p),detail::output_span(out));return out;
}
template<class T> FitResult make_fit_result(std::size_t g) {
    return {anndatapp::Array::zeros<T>({g}),anndatapp::Array::zeros<T>({g}),anndatapp::Array::zeros<T>({g}),anndatapp::Array::zeros<bool>({g}),anndatapp::Array::zeros<std::uint8_t>({g})};
}
template<class T> FitBuffers<T> fit_buffers(FitResult& r) {
    auto g=r.gamma.extent(0);
    for(auto* a:{&r.gamma,&r.offset,&r.r2,&r.genes,&r.status})
        if(a->rank()!=1||a->extent(0)!=g)throw std::invalid_argument("fit diagnostics must have matching gene extents");
    return {g,r.gamma.edit().dense<T>().contiguous_data(),r.offset.edit().dense<T>().contiguous_data(),r.r2.edit().dense<T>().contiguous_data(),r.genes.edit().dense<bool>().contiguous_data(),r.status.edit().dense<std::uint8_t>().contiguous_data()};
}
template<class T> StochasticFitResult make_stochastic_fit_result(std::size_t g) {
    auto make=[g]{return anndatapp::Array::zeros<T>({g});};
    return {make_fit_result<T>(g),make(),make(),make(),make(),make(),anndatapp::Array::zeros<std::uint8_t>({g}),make(),false};
}
template<class T> StochasticFitBuffers<T> fit_buffers(StochasticFitResult& r) {
    auto a=fit_buffers<T>(r.initialization);auto g=a.size;
    for(auto* v:{&r.gamma,&r.offset,&r.offset2,&r.first_scale,&r.second_scale,&r.status,&r.objective})
        if(v->rank()!=1||v->extent(0)!=g)throw std::invalid_argument("stochastic diagnostics must have matching gene extents");
    auto ptr=[](anndatapp::Array& v){return v.edit().dense<T>().contiguous_data();};
    return {a,ptr(r.gamma),ptr(r.offset),ptr(r.offset2),ptr(r.first_scale),ptr(r.second_scale),r.status.edit().dense<std::uint8_t>().contiguous_data(),ptr(r.objective)};
}
namespace reference {
template<class T> BasicMomentBlock<T> moments(const PreparedInput& p,Workspace& workspace,GeneRange range) {
    const auto& s=detail::Access::get(p);auto& w=detail::Access::get(workspace).get<T>();
    if(w.owner!=&s)throw std::invalid_argument("workspace belongs to another input");detail::check_range(s,range,w.block);
    if(range.count)s.input.moments_selected(s.genes.data()+range.first,range.count,w.ms.data(),w.mu.data());
    return {range.first,range.count,w.n,std::ptrdiff_t(range.count),w.ms.data(),w.mu.data()};
}
template<class T> BasicFitBlock<T> fit(const PreparedInput& p,Workspace& workspace,BasicMomentBlock<T> m,FitBuffers<T> b,FitOptions o) {
    const auto& s=detail::Access::get(p);auto& w=detail::Access::get(workspace).get<T>();
    if(w.owner!=&s)throw std::invalid_argument("workspace belongs to another input");return detail::fit_range(s,w,m,b,o);
}
template<class T> void replace_moments(const PreparedInput& p,Workspace& workspace,BasicMomentBlock<T>& m,std::size_t offset,std::size_t count,NumericView a,NumericView b) {
    const auto& s=detail::Access::get(p);auto& w=detail::Access::get(workspace).get<T>();
    if(w.owner!=&s)throw std::invalid_argument("workspace belongs to another input");
    detail::check_range(s,{m.first,m.count},w.block);detail::check_replacement<T>(m,offset,count,a,b);
    if(count&&m.rows)s.input.moments_from(a,b,m.spliced+offset,m.unspliced+offset,std::size_t(m.stride),s.checked);
}
template<class T> bool finalize_selection(const PreparedInput& p,FitBuffers<T> b,FitOptions o,T* score,std::size_t capacity) {
    return detail::select_genes(detail::Access::get(p),b,o,score,capacity);
}
template<class T> BasicMomentBlock<T> initial_moments(const PreparedInput& p,StochasticWorkspace& workspace,GeneRange range) {
    const auto& s=detail::Access::get(p);auto& w=detail::Access::get(workspace).get<T>();
    if(w.owner!=&s)throw std::invalid_argument("workspace belongs to another input");detail::check_range(s,range,w.block);
    if(range.count)s.input.moments_selected(s.genes.data()+range.first,range.count,w.ms.data(),w.mu.data());
    return {range.first,range.count,w.n,std::ptrdiff_t(range.count),w.ms.data(),w.mu.data()};
}
template<class T> BasicFitBlock<T> fit(const PreparedInput& p,StochasticWorkspace& workspace,BasicMomentBlock<T> m,FitBuffers<T> b,FitOptions o) {
    const auto& s=detail::Access::get(p);auto& w=detail::Access::get(workspace).get<T>();
    if(w.owner!=&s)throw std::invalid_argument("workspace belongs to another input");return detail::fit_range(s,w,m,b,o);
}
template<class T> void replace_moments(const PreparedInput& p,StochasticWorkspace& workspace,BasicMomentBlock<T>& m,std::size_t offset,std::size_t count,NumericView a,NumericView b) {
    const auto& s=detail::Access::get(p);auto& w=detail::Access::get(workspace).get<T>();
    if(w.owner!=&s)throw std::invalid_argument("workspace belongs to another input");
    detail::check_range(s,{m.first,m.count},w.block);detail::check_replacement<T>(m,offset,count,a,b);
    if(count&&m.rows)s.input.moments_from(a,b,m.spliced+offset,m.unspliced+offset,std::size_t(m.stride),s.checked);
}
template<class T> void replace_moments(const PreparedInput& p,StochasticWorkspace& workspace,BasicStochasticMomentBlock<T>& m,std::size_t offset,std::size_t count,NumericView a,NumericView b) {
    const auto& s=detail::Access::get(p);auto& w=detail::Access::get(workspace).get<T>();
    if(w.owner!=&s)throw std::invalid_argument("workspace belongs to another input");
    detail::check_range(s,{m.first,m.count},w.block);detail::check_replacement<T>(m,offset,count,a,b);
    if(!count||!m.rows)return;
    for(auto* scratch:{w.source_s.data(),w.source_u.data()}) {
        auto target=detail::contiguous_span(reinterpret_cast<const std::byte*>(scratch),detail::mul(m.rows,sizeof(T)));
        if(detail::aliases(a,target)||detail::aliases(b,target))throw std::invalid_argument("alternate source aliases source scratch");
    }
    s.input.second_moments_from(a,b,m.spliced+offset,m.unspliced+offset,m.spliced_second+offset,m.cross+offset,m.unspliced_second?m.unspliced_second+offset:nullptr,w.source_s.data(),w.source_u.data(),std::size_t(m.stride),s.checked);
}
template<class T> SelectionResult finalize_selection(const PreparedInput& p,StochasticFitBuffers<T> b,StochasticOptions o,T* score,std::size_t capacity) {
    return detail::select_genes(detail::Access::get(p),b,o,score,capacity);
}
template<class T> BasicStochasticMomentBlock<T> second_moments(const PreparedInput& p,StochasticWorkspace& workspace,GeneRange range) {
    const auto& s=detail::Access::get(p);auto& w=detail::Access::get(workspace).get<T>();
    if(w.owner!=&s)throw std::invalid_argument("workspace belongs to another input");detail::check_range(s,range,w.block);
    if(range.count)s.input.second_moments_selected(s.genes.data()+range.first,range.count,w.ms.data(),w.mu.data(),w.mss.data(),w.mus.data(),w.options.unspliced_second?w.muu.data():nullptr,w.source_s.data(),w.source_u.data());
    return {range.first,range.count,w.n,std::ptrdiff_t(range.count),w.ms.data(),w.mu.data(),w.mss.data(),w.mus.data(),w.options.unspliced_second?w.muu.data():nullptr};
}
template<class T> BasicStochasticScaleBlock<T> scales(const PreparedInput& p,StochasticWorkspace& workspace,BasicStochasticMomentBlock<T> m,StochasticFitBuffers<T> b,StochasticOptions o,SelectionResult selection) {
    const auto& s=detail::Access::get(p);auto& w=detail::Access::get(workspace).get<T>();
    if(w.owner!=&s)throw std::invalid_argument("workspace belongs to another input");return detail::scales_range(s,w,m,b,o,selection);
}
template<class T> BasicStochasticFitBlock<T> fit(const PreparedInput& p,StochasticWorkspace& workspace,BasicStochasticMomentBlock<T> m,BasicStochasticScaleBlock<T> scales,StochasticFitBuffers<T> b,StochasticOptions o,SelectionResult selection) {
    const auto& s=detail::Access::get(p);auto& w=detail::Access::get(workspace).get<T>();
    if(w.owner!=&s)throw std::invalid_argument("workspace belongs to another input");return detail::fit_range(s,w,m,scales,b,o,selection);
}
template<class T> void write_residuals(BasicMomentBlock<T> m,BasicFitBlock<T> f,DenseOutput out) {
    if(out.rows!=m.rows||m.first>out.columns||m.count>out.columns-m.first||f.first!=m.first||f.count!=m.count)throw std::invalid_argument("incompatible residual output block");
    detail::output_dispatch(out.type,[&]<class O>() {
        for(std::size_t i=0;i<m.rows;++i)for(std::size_t j=0;j<m.count;++j) {
            auto q=std::ptrdiff_t(i)*m.stride+std::ptrdiff_t(j);
            O value=detail::output_cast<O>(m.unspliced[q]-f.gamma[j]*m.spliced[q]-f.offset[j]);
            std::memcpy(out.data+std::ptrdiff_t(i)*out.row_stride+std::ptrdiff_t(m.first+j)*out.column_stride,&value,sizeof value);
        }
    });
}
template<class T> void write_residuals(BasicStochasticMomentBlock<T> m,BasicStochasticFitBlock<T> f,DenseOutput out,std::optional<DenseOutput> second) {
    auto write=[&](DenseOutput dst,bool residual2) {
        if(dst.rows!=m.rows||m.first>dst.columns||m.count>dst.columns-m.first||f.first!=m.first||f.count!=m.count)throw std::invalid_argument("incompatible stochastic residual output block");
        detail::output_dispatch(dst.type,[&]<class O>() {
            for(std::size_t i=0;i<m.rows;++i)for(std::size_t j=0;j<m.count;++j) {
                auto q=std::ptrdiff_t(i)*m.stride+std::ptrdiff_t(j);T x=m.spliced[q],y=m.unspliced[q];
                T value=residual2 ? (f.status[j]==std::uint8_t(StochasticStatus::ok) ? (2*m.cross[q]+y-2*x*y)-f.gamma[j]*(2*m.spliced_second[q]-x-2*x*x)+2*f.offset[j]*x-f.offset2[j] : 0) : y-f.gamma[j]*x-f.offset[j];
                O v=detail::output_cast<O>(value);std::memcpy(dst.data+std::ptrdiff_t(i)*dst.row_stride+std::ptrdiff_t(m.first+j)*dst.column_stride,&v,sizeof v);
            }
        });
    };
    write(out,false);if(second)write(*second,true);
}
}
#define VELOCITY_BLOCK_INSTANTIATE(T) \
template FitResult make_fit_result<T>(std::size_t); \
template FitBuffers<T> fit_buffers<T>(FitResult&); \
template StochasticFitResult make_stochastic_fit_result<T>(std::size_t); \
template StochasticFitBuffers<T> fit_buffers<T>(StochasticFitResult&); \
template BasicMomentBlock<T> reference::moments<T>(const PreparedInput&,Workspace&,GeneRange); \
template BasicFitBlock<T> reference::fit<T>(const PreparedInput&,Workspace&,BasicMomentBlock<T>,FitBuffers<T>,FitOptions); \
template void reference::replace_moments<T>(const PreparedInput&,Workspace&,BasicMomentBlock<T>&,std::size_t,std::size_t,NumericView,NumericView); \
template void reference::replace_moments<T>(const PreparedInput&,StochasticWorkspace&,BasicMomentBlock<T>&,std::size_t,std::size_t,NumericView,NumericView); \
template void reference::replace_moments<T>(const PreparedInput&,StochasticWorkspace&,BasicStochasticMomentBlock<T>&,std::size_t,std::size_t,NumericView,NumericView); \
template bool reference::finalize_selection<T>(const PreparedInput&,FitBuffers<T>,FitOptions,T*,std::size_t); \
template BasicMomentBlock<T> reference::initial_moments<T>(const PreparedInput&,StochasticWorkspace&,GeneRange); \
template BasicFitBlock<T> reference::fit<T>(const PreparedInput&,StochasticWorkspace&,BasicMomentBlock<T>,FitBuffers<T>,FitOptions); \
template SelectionResult reference::finalize_selection<T>(const PreparedInput&,StochasticFitBuffers<T>,StochasticOptions,T*,std::size_t); \
template BasicStochasticMomentBlock<T> reference::second_moments<T>(const PreparedInput&,StochasticWorkspace&,GeneRange); \
template BasicStochasticScaleBlock<T> reference::scales<T>(const PreparedInput&,StochasticWorkspace&,BasicStochasticMomentBlock<T>,StochasticFitBuffers<T>,StochasticOptions,SelectionResult); \
template BasicStochasticFitBlock<T> reference::fit<T>(const PreparedInput&,StochasticWorkspace&,BasicStochasticMomentBlock<T>,BasicStochasticScaleBlock<T>,StochasticFitBuffers<T>,StochasticOptions,SelectionResult); \
template void reference::write_residuals<T>(BasicMomentBlock<T>,BasicFitBlock<T>,DenseOutput); \
template void reference::write_residuals<T>(BasicStochasticMomentBlock<T>,BasicStochasticFitBlock<T>,DenseOutput,std::optional<DenseOutput>);
VELOCITY_BLOCK_INSTANTIATE(float)
VELOCITY_BLOCK_INSTANTIATE(double)
#undef VELOCITY_BLOCK_INSTANTIATE
}
