module;
#include "cuda/bridge.hh"
#include "cuda/sensitivity_bridge.hh"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

module velocity.cuda;

namespace velocity::cuda {
namespace ac = anndata::cuda;
namespace bridge = ::velocity_cuda_bridge;
using ac::Access;
using ac::DeviceBuffer;
using ac::ScalarType;
using ac::Span;

namespace detail {
namespace {
void check(int code) { if(code)throw std::runtime_error(std::string("velocity CUDA: ")+cudaGetErrorString(cudaError_t(code))); }
struct Guard {
    int previous;
    explicit Guard(int device) {check(cudaGetDevice(&previous));if(previous!=device)check(cudaSetDevice(device));}
    ~Guard() {if(previous>=0)cudaSetDevice(previous);}
};
std::size_t add(std::size_t a,std::size_t b) {if(b>SIZE_MAX-a)throw std::length_error("CUDA storage size overflow");return a+b;}
std::size_t mul(std::size_t a,std::size_t b) {if(a && b>SIZE_MAX/a)throw std::length_error("CUDA storage size overflow");return a*b;}
bool floating(ScalarType t) {return t==ScalarType::float16 || t==ScalarType::float32 || t==ScalarType::float64;}
ScalarType dtype(ComputePrecision t) {
    if(t!=ComputePrecision::float32 && t!=ComputePrecision::float64)throw std::invalid_argument("invalid CUDA compute precision");
    return t==ComputePrecision::float32?ScalarType::float32:ScalarType::float64;
}
std::size_t rows(MatrixView m) {return m.sparse?m.compressed.rows:m.dense.rows;}
std::size_t columns(MatrixView m) {return m.sparse?m.compressed.columns:m.dense.columns;}
int device(MatrixView m) {return m.sparse?m.compressed.values.device:m.dense.values.device;}
bridge::Matrix matrix(MatrixView m) {return {m.sparse?1u:0u,m.dense,m.compressed};}
MatrixView matrix_view(bridge::Matrix m) {return m.sparse?MatrixView(m.compressed):MatrixView(m.dense);}
Stream stream_for(Stream stream,int device) {
    if(stream.device<0)stream.device=device;
    if(stream.device!=device)throw std::invalid_argument("stream and data devices differ");
    return stream;
}
void validate_matrix(MatrixView m,int expected) {
    if(m.sparse)ac::validate(m.compressed);else ac::validate(m.dense);
    auto type=m.sparse?m.compressed.dtype:m.dense.dtype;
    if(!floating(type))throw std::invalid_argument("velocity matrices must have floating storage");
    if(device(m)!=expected)throw std::invalid_argument("velocity input devices differ");
}
void validate_mask(DenseView v,std::size_t n,int device) {
    ac::validate(v);
    if(v.rows!=n || v.columns!=1 || v.values.device!=device || (v.dtype!=ScalarType::boolean && v.dtype!=ScalarType::uint8))
        throw std::invalid_argument("velocity device mask must be a matching boolean/uint8 column");
}
void validate_values(MatrixView m,Stream stream,bool contents) {
    DeviceBuffer error(sizeof(std::uint32_t),stream.device);
    ac::zero(error.span(),0,error.size_bytes(),stream);
    if(contents && m.sparse)ac::validate_sparse_contents(m.compressed,error.span(),stream);
    Guard guard(stream.device);check(bridge::validate(matrix(m),false,static_cast<int*>(error.data()),stream.native));
    std::uint32_t status=0;ac::copy_to_host(&status,error.span(),0,sizeof status,stream);ac::synchronize(stream);
    if(status)throw std::invalid_argument(status&3?"invalid device sparse contents":"device input contains nonfinite values");
}
struct MatrixStorage {
    bridge::PreparedMatrix view{};
    DeviceBuffer offsets,edges;
    std::size_t bytes() const {return offsets.size_bytes()+edges.size_bytes();}
};
MatrixStorage prepare_matrix(MatrixView m,bool by_rows,Stream stream) {
    MatrixStorage out;out.view.matrix=matrix(m);
    if(!m.sparse)return out;
    auto major=by_rows?m.compressed.rows:m.compressed.columns;
    out.offsets=DeviceBuffer(mul(add(major,1),sizeof(std::uint64_t)),stream.device);
    out.edges=DeviceBuffer(mul(m.compressed.nnz,sizeof(bridge::Edge)),stream.device);
    DeviceBuffer counts(mul(add(major,1),sizeof(std::uint64_t)),stream.device);
    std::size_t bytes=0;Guard guard(stream.device);check(bridge::prepare_scratch(major,&bytes));
    DeviceBuffer scratch(bytes,stream.device);
    check(bridge::prepare(matrix(m),by_rows,static_cast<std::uint64_t*>(out.offsets.data()),
        static_cast<bridge::Edge*>(out.edges.data()),static_cast<std::uint64_t*>(counts.data()),scratch.data(),bytes,stream.native));
    // Temporary preparation storage is released here, explicitly outside the
    // numerical loop; complete its last use before allocator destruction.
    ac::synchronize(stream);
    out.view.offsets=static_cast<const std::uint64_t*>(out.offsets.data());out.view.edges=static_cast<const bridge::Edge*>(out.edges.data());
    return out;
}
void* pointer(DenseView v) {return v.values.data?static_cast<char*>(v.values.data)+v.origin:nullptr;}
DenseView view(Span span,std::size_t origin,std::size_t n,std::size_t g,ScalarType type,bool column_major=false) {
    return {span,origin,n,g,column_major?1:std::int64_t(g),column_major?std::int64_t(n):1,type};
}
struct Range {std::uintptr_t begin=0,end=0;};
Range range(DenseView v) {
    if(!v.values.data || !v.rows || !v.columns)return {};
    auto row=__int128(v.rows-1)*v.row_stride,col=__int128(v.columns-1)*v.column_stride;
    auto lo=std::min(__int128(0),row)+std::min(__int128(0),col);
    auto hi=std::max(__int128(0),row)+std::max(__int128(0),col);
    auto width=ac::scalar_size(v.dtype);auto base=reinterpret_cast<std::uintptr_t>(v.values.data)+v.origin;
    return {std::uintptr_t(__int128(base)+lo*width),std::uintptr_t(__int128(base)+hi*width+width)};
}
Range range(Span p) {auto b=reinterpret_cast<std::uintptr_t>(p.data);return p.bytes?Range{b,b+p.bytes}:Range{};}
bool overlaps(Range a,Range b) {return a.begin<b.end && b.begin<a.end;}
bool separate_elements(DenseView v) {
    if(!v.rows || !v.columns)return true;
    auto magnitude=[](std::int64_t x) {return x>=0?std::uint64_t(x):std::uint64_t(-(x+1))+1;};
    auto a=magnitude(v.row_stride),b=magnitude(v.column_stride);
    return !((v.rows>1 && !a)||(v.columns>1 && !b)
        ||(v.rows>1 && v.columns>1 && v.rows>b/std::gcd(a,b) && v.columns>a/std::gcd(a,b)));
}
void disjoint(Range target,MatrixView m) {
    if(!m.sparse) {if(overlaps(target,range(m.dense)))throw std::invalid_argument("device output aliases input");}
    else for(auto p:{m.compressed.values,m.compressed.indices,m.compressed.offsets})
        if(overlaps(target,range(p)))throw std::invalid_argument("device output aliases sparse input");
}
void check_fit_options(const FitOptions& o) {
    if(o.trimming!=Trimming::none && o.trimming!=Trimming::upper && o.trimming!=Trimming::tails)
        throw std::invalid_argument("invalid trimming mode");
    if(o.low_signal!=LowSignal::keep_filters && o.low_signal!=LowSignal::scvelo_percentile)
        throw std::invalid_argument("invalid low-signal policy");
    if(!std::isfinite(o.percentile_low)||!std::isfinite(o.percentile_high)||o.percentile_low<0||o.percentile_high>100||o.percentile_low>o.percentile_high
       ||std::isnan(o.ratio_min)||std::isnan(o.ratio_max)||o.ratio_min>o.ratio_max||!std::isfinite(o.min_r2)||!std::isfinite(o.min_ratio))
        throw std::invalid_argument("invalid fit options");
}
bridge::Options options(StochasticOptions o) {
    check_fit_options(o.initialization);auto& f=o.initialization;
    return {f.percentile_low,f.percentile_high,f.ratio_min,f.ratio_max,f.min_r2,f.min_ratio,
        std::uint32_t(f.trimming),f.fit_intercept,f.positive_intercept,f.r2_adjusted,
        std::uint32_t(f.low_signal),o.fit_second_intercept,o.refit_all_if_none};
}
}

struct PreparedState {
    Input input;
    MatrixStorage graph,spliced,unspliced;
    std::vector<std::uint64_t> host_genes;
    DeviceBuffer genes;
    std::size_t n=0,g=0,source_g=0;
    int device=0;
    bool checked=false;
};
struct ExpressionState {
    const PreparedState* owner=nullptr;
    MatrixStorage spliced,unspliced;
    BlockSource source;
    bool available=false;
};
struct WorkspaceState {
    const PreparedState* owner;
    DeviceBuffer storage;
    bridge::Fields fields;
    std::size_t block=0,elements=0;
    std::size_t budget=0,staging=0;
    ComputePrecision precision;
    bool stochastic=false,unspliced_second=false;
};
struct ResultState {DeviceBuffer numeric,flags;ResultView view;};
struct StageState {
    const PreparedState* p;
    WorkspaceState* w;
    ResultView result;
    bridge::Result raw;
    bridge::Options options;
    Outputs outputs;
    BlockSink sink;
    Stream stream;
    std::size_t next=0;
    bool stochastic=false,resolved=false,finished=false;
    const ExpressionState* original=nullptr;
    std::vector<std::uint8_t> promotions;
    std::vector<std::uint64_t> promoted_genes,promoted_columns;
};
struct Access {
    static const PreparedState& get(const PreparedInput& p) {if(!p.state_)throw std::invalid_argument("moved device prepared input");return *p.state_;}
    static WorkspaceState& get(Workspace& w) {if(!w.state_)throw std::invalid_argument("moved device workspace");return *w.state_;}
    static const ExpressionState& get(const ExpressionSource& p) {if(!p.state_)throw std::invalid_argument("moved expression source");return *p.state_;}
};
namespace {
std::unique_ptr<WorkspaceState> workspace(const PreparedInput& p,std::size_t budget,std::size_t block,
    ComputePrecision precision,bool stochastic,bool muu) {
    auto& ps=Access::get(p);
    if(precision==ComputePrecision::automatic) {
        auto wt=ps.input.weights.sparse?ps.input.weights.compressed.dtype:ps.input.weights.dense.dtype;
        auto st=ps.input.source.acquire?ps.input.source_spliced_dtype:(ps.input.spliced.sparse?ps.input.spliced.compressed.dtype:ps.input.spliced.dense.dtype);
        auto ut=ps.input.source.acquire?ps.input.source_unspliced_dtype:(ps.input.unspliced.sparse?ps.input.unspliced.compressed.dtype:ps.input.unspliced.dense.dtype);
        bool half=wt==ScalarType::float16 || st==ScalarType::float16 || ut==ScalarType::float16;
        bool double_precision=wt==ScalarType::float64 || st==ScalarType::float64 || ut==ScalarType::float64;
        precision=half && !double_precision?ComputePrecision::float32:ComputePrecision::float64;
    }
    auto type=dtype(precision);
    auto original_budget=budget;
    auto staging=ps.input.source.acquire?ps.input.source.staging_bytes:0;
    if(staging>budget)throw std::length_error("base block-source staging exceeds workspace budget");
    budget-=staging;
    if(!block)throw std::invalid_argument("CUDA block_genes must be positive");
    block=std::min({block,std::size_t(64),std::max(ps.g,std::size_t(1))});
    auto slots=std::size_t(stochastic?(muu?8:7):5),width=ac::scalar_size(type);
    auto mask=ps.input.fit_mask?ps.n:0;
    auto per_gene=mul(mul(ps.n,slots),width);
    if(budget<mask || (per_gene && per_gene>budget-mask))
        throw std::length_error("CUDA workspace cannot fit one gene");
    if(per_gene)block=std::min(block,(budget-mask)/per_gene);
    auto out=std::make_unique<WorkspaceState>();out->owner=&ps;out->precision=precision;
    out->budget=original_budget;out->staging=staging;
    out->block=block;out->elements=mul(ps.n,block);out->stochastic=stochastic;out->unspliced_second=muu;
    auto numerical=mul(mul(out->elements,slots),width);
    auto bytes=add(numerical,mask);
    // The empty-cell path still supplies a valid, never-dereferenced scratch
    // pointer to the scalar fitting primitive.
    if(!bytes)bytes=1;
    if(bytes>budget)throw std::length_error("CUDA workspace budget exhausted");
    out->storage=DeviceBuffer(bytes,ps.device);auto* base=static_cast<char*>(out->storage.data());std::size_t offset=0;
    auto field=[&]() {void* p=base+offset;offset+=out->elements*width;return p;};
    out->fields.source_s=field();out->fields.source_u=field();out->fields.ms=field();out->fields.mu=field();
    if(stochastic) {out->fields.mss=field();out->fields.mus=field();if(muu)out->fields.muu=field();}
    out->fields.scores=field();out->fields.mask=mask?reinterpret_cast<std::uint8_t*>(base+numerical):nullptr;
    return out;
}
DenseView field_view(const WorkspaceState& w,void* field,std::size_t count) {
    if(!field)return {};
    auto origin=std::size_t(static_cast<char*>(field)-static_cast<char*>(w.storage.data()));
    return view(w.storage.span(),origin,w.owner->n,count,dtype(w.precision),true);
}
MomentBlock block_view(const WorkspaceState& w,std::size_t first,std::size_t count,bool second) {
    return {first,count,field_view(w,w.fields.ms,count),field_view(w,w.fields.mu,count),
        second?field_view(w,w.fields.mss,count):DenseView{},second?field_view(w,w.fields.mus,count):DenseView{},
        second?field_view(w,w.fields.muu,count):DenseView{}};
}
void check_vector(DenseView v,std::size_t g,ScalarType type,int device) {
    ac::validate(v,true);
    if(v.rows!=g || v.columns!=1 || v.dtype!=type || v.row_stride!=1 || v.values.device!=device)
        throw std::invalid_argument("device diagnostic shape/type/stride mismatch");
}
bridge::Result result(ResultView v,const PreparedState& p,ComputePrecision precision,bool stochastic) {
    auto type=dtype(precision);
    for(auto field:{v.initial_gamma,v.initial_offset,v.r2,v.gamma,v.offset})check_vector(field,p.g,type,p.device);
    for(auto field:{v.selected,v.initial_status,v.status})check_vector(field,p.g,ScalarType::uint8,p.device);
    for(auto field:{v.low_signal,v.refit_all})check_vector(field,1,ScalarType::uint8,p.device);
    if(stochastic)for(auto field:{v.offset2,v.first_scale,v.second_scale,v.objective})check_vector(field,p.g,type,p.device);
    return {pointer(v.initial_gamma),pointer(v.initial_offset),pointer(v.r2),static_cast<std::uint8_t*>(pointer(v.selected)),
        static_cast<std::uint8_t*>(pointer(v.initial_status)),pointer(v.gamma),pointer(v.offset),pointer(v.offset2),
        pointer(v.first_scale),pointer(v.second_scale),pointer(v.objective),static_cast<std::uint8_t*>(pointer(v.status)),
        static_cast<std::uint8_t*>(pointer(v.low_signal)),static_cast<std::uint8_t*>(pointer(v.refit_all))};
}
std::array<DenseView,14> result_fields(ResultView v) {
    return {v.initial_gamma,v.initial_offset,v.r2,v.selected,v.initial_status,v.gamma,v.offset,v.offset2,
        v.first_scale,v.second_scale,v.objective,v.status,v.low_signal,v.refit_all};
}
void check_targets(const PreparedState& p,const WorkspaceState& w,ResultView r,Outputs out) {
    std::array<DenseView,7> fields{out.velocity,out.second_residual,out.spliced,out.unspliced,out.spliced_second,out.cross,out.unspliced_second};
    for(std::size_t i=0;i<fields.size();++i) {
        auto v=fields[i];if(!v.values.data)continue;ac::validate(v,true);
        if(v.rows!=p.n || v.columns!=(out.block_local?w.block:p.g) || !floating(v.dtype) || v.values.device!=p.device)
            throw std::invalid_argument("device output shape/type mismatch");
        if(!separate_elements(v))
            throw std::invalid_argument("device output logical elements overlap");
        if((i==1 || i>=4) && !w.stochastic)throw std::invalid_argument("second-order output needs stochastic workspace");
        if(i==6 && !w.unspliced_second)throw std::invalid_argument("unspliced second moment not allocated");
        auto target=range(v);disjoint(target,p.input.weights);
        if(!p.input.source.acquire) {disjoint(target,p.input.spliced);disjoint(target,p.input.unspliced);}
        if(overlaps(target,range(w.storage.span())))throw std::invalid_argument("device output aliases workspace");
        if((p.input.fit_mask && overlaps(target,range(*p.input.fit_mask))) || (p.input.gene_mask && overlaps(target,range(*p.input.gene_mask))))
            throw std::invalid_argument("device output aliases mask");
        for(auto field:result_fields(r))if(overlaps(target,range(field)))throw std::invalid_argument("device output aliases diagnostics");
        for(std::size_t j=0;j<i;++j)if(overlaps(target,range(fields[j])))throw std::invalid_argument("device outputs overlap");
    }
    for(auto field:result_fields(r))if(field.values.data) {
        auto target=range(field);disjoint(target,p.input.weights);
        if(!p.input.source.acquire) {disjoint(target,p.input.spliced);disjoint(target,p.input.unspliced);}
        if(overlaps(target,range(w.storage.span())))throw std::invalid_argument("device diagnostics alias workspace");
        if((p.input.fit_mask && overlaps(target,range(*p.input.fit_mask))) || (p.input.gene_mask && overlaps(target,range(*p.input.gene_mask))))
            throw std::invalid_argument("device diagnostics alias mask");
    }
    auto diagnostics=result_fields(r);
    for(std::size_t i=0;i<diagnostics.size();++i)for(std::size_t j=0;j<i;++j)
        if(overlaps(range(diagnostics[i]),range(diagnostics[j]))) {
            bool alias=!w.stochastic && ((i==5 && j==0)||(i==6 && j==1)||(i==11 && j==4));
            if(!alias || pointer(diagnostics[i])!=pointer(diagnostics[j]))throw std::invalid_argument("device diagnostic arrays overlap");
        }
}
std::unique_ptr<StageState> stages(const PreparedInput& input,Workspace& workspace,ResultView r,
    StochasticOptions o,Outputs outputs,Stream stream,bool stochastic,BlockSink sink) {
    auto& p=Access::get(input);auto& w=Access::get(workspace);
    if(w.owner!=&p || w.stochastic!=stochastic)throw std::invalid_argument("CUDA workspace belongs to a different input/method");
    auto out=std::make_unique<StageState>();out->p=&p;out->w=&w;out->result=r;
    out->raw=result(r,p,w.precision,stochastic);out->options=options(o);out->outputs=outputs;
    out->stream=stream_for(stream,p.device);out->stochastic=stochastic;out->resolved=!stochastic;
    out->sink=sink;
    check_targets(p,w,r,outputs);return out;
}
void check_range(StageState& s,std::size_t first,std::size_t count,bool initialization) {
    if(s.finished || !count || count>s.w->block || first!=s.next || first>s.p->g || count>s.p->g-first
        ||(s.stochastic && initialization==s.resolved))throw std::invalid_argument("invalid CUDA block stage order/range");
}
void check_fit_block(StageState& s,MomentBlock m,FitBlock f) {
    if(f.first!=m.first || f.count!=m.count)
        throw std::invalid_argument("fit block must describe the stage result");
    auto expected=result_fields(s.result),actual=result_fields(f.result);
    for(std::size_t i=0;i<expected.size();++i) {
        auto a=actual[i],b=expected[i];
        if(a.values.data!=b.values.data || a.values.bytes!=b.values.bytes || a.values.device!=b.values.device || a.values.access!=b.values.access
           ||a.origin!=b.origin || a.rows!=b.rows || a.columns!=b.columns || a.row_stride!=b.row_stride || a.column_stride!=b.column_stride || a.dtype!=b.dtype)
            throw std::invalid_argument("fit block must describe the stage result");
    }
}
bridge::Fields moment_fields(StageState& s,MomentBlock m,bool second) {
    check_range(s,m.first,m.count,!s.resolved);
    auto fields=s.w->fields;auto acquire=[&](DenseView v) {
        ac::validate(v);
        if(v.rows!=s.p->n || v.columns!=m.count || v.dtype!=dtype(s.w->precision) || v.row_stride!=1
            || v.column_stride!=std::int64_t(s.p->n) || v.values.device!=s.p->device)
            throw std::invalid_argument("CUDA moment block must be column-major with compute dtype");
        auto source=range(v);
        auto score=range(field_view(*s.w,s.w->fields.scores,s.w->block));
        auto mask_begin=reinterpret_cast<std::uintptr_t>(s.w->fields.mask);
        if(overlaps(source,score) || (s.w->fields.mask && overlaps(source,{mask_begin,mask_begin+s.p->n})))
            throw std::invalid_argument("device moments alias writable score/mask scratch");
        for(auto field:result_fields(s.result))if(overlaps(source,range(field)))throw std::invalid_argument("device moments alias diagnostics");
        auto& out=s.outputs;
        for(auto field:{out.velocity,out.second_residual,out.spliced,out.unspliced,out.spliced_second,out.cross,out.unspliced_second})
            if(overlaps(source,range(field)))throw std::invalid_argument("device moments alias output");
        return pointer(v);
    };
    fields.ms=acquire(m.spliced);fields.mu=acquire(m.unspliced);
    if(second) {fields.mss=acquire(m.spliced_second);fields.mus=acquire(m.cross);}
    if(m.unspliced_second.values.data)fields.muu=acquire(m.unspliced_second);
    return fields;
}
void check_source_block(SourceBlock block,std::size_t rows,std::size_t columns,int device) {
    for(auto v:{block.spliced,block.unspliced}) {
        ac::validate(v);
        if(v.rows!=rows || v.columns!=columns || !floating(v.dtype) || v.values.device!=device)
            throw std::invalid_argument("block source returned an incompatible dense device view");
    }
}
void check_source_aliases(const StageState& s,SourceBlock source) {
    auto& out=s.outputs;
    for(auto v:{source.spliced,source.unspliced}) {
        auto input=range(v);
        if(overlaps(input,range(s.w->storage.span())))throw std::invalid_argument("block source aliases workspace");
        for(auto field:result_fields(s.result))if(overlaps(input,range(field)))throw std::invalid_argument("block source aliases diagnostics");
        for(auto field:{out.velocity,out.second_residual,out.spliced,out.unspliced,out.spliced_second,out.cross,out.unspliced_second})
            if(overlaps(input,range(field)))throw std::invalid_argument("block source aliases output");
    }
}
void pack_base(StageState& s,std::size_t first,std::size_t count) {
    auto& p=*s.p;auto& w=*s.w;auto* genes=static_cast<const std::uint64_t*>(p.genes.data());
    if(p.input.source.acquire) {
        SourceBlock block;p.input.source.acquire(p.input.source.context,p.host_genes.data()+first,count,s.stream,block);
        check(cudaSetDevice(p.device));
        check_source_block(block,p.n,count,p.device);
        check_source_aliases(s,block);
        if(block.spliced.dtype!=p.input.source_spliced_dtype || block.unspliced.dtype!=p.input.source_unspliced_dtype)
            throw std::invalid_argument("block source dtype differs from its declaration");
        if(block.available)for(std::size_t j=0;j<count;++j)if(!block.available[j])throw std::invalid_argument("base block source unavailable");
        check(bridge::pack({matrix(MatrixView(block.spliced))},{matrix(MatrixView(block.unspliced))},nullptr,0,count,p.n,w.fields,
            w.precision==ComputePrecision::float32,s.stream.native));
    } else check(bridge::pack(p.spliced.view,p.unspliced.view,genes,first,count,p.n,w.fields,
        w.precision==ComputePrecision::float32,s.stream.native));
}
MomentBlock calculate(StageState& s,std::size_t first,std::size_t count,bool initialization,
    const ExpressionState* original=nullptr,const std::uint8_t* promotion=nullptr,std::uint8_t* applied=nullptr) {
    check_range(s,first,count,initialization);
    if(s.stochastic && initialization && !s.promotions.empty())
        std::fill_n(s.promotions.data()+first,count,std::uint8_t(0));
    if(s.stochastic && !initialization) {
        if(!s.promotions.empty()) {
            if(original && original!=s.original)throw std::invalid_argument("alternate source changed during stochastic run");
            original=s.original;
        } else {
            if(promotion)for(std::size_t j=0;j<count;++j)if(promotion[s.p->host_genes[first+j]])
                throw std::invalid_argument("promotion must be chosen before global initialization selection");
            original=nullptr;
        }
    } else if(!promotion)original=nullptr;
    if(original && original->source.acquire) {
        bool requested=false;
        for(std::size_t j=0;j<count;++j)requested|=s.stochastic && !initialization
            ? s.promotions[first+j]==1 : (promotion && promotion[s.p->host_genes[first+j]]);
        if(requested && (original->source.staging_bytes>s.w->budget-s.w->staging
            ||s.w->storage.size_bytes()>s.w->budget-s.w->staging-original->source.staging_bytes))
            throw std::length_error("original-source staging exceeds aggregate workspace budget");
    }
    if(original && s.stochastic && initialization) {
        if(s.original && original!=s.original)throw std::invalid_argument("alternate source changed during initialization");
        s.original=original;
        if(s.promotions.empty())s.promotions.resize(s.p->g,0);
    }
    if(original && original->available && !original->source.acquire) {
        for(auto m:{original->spliced.view.matrix,original->unspliced.view.matrix}) {
            auto source=matrix_view(m);disjoint(range(s.w->storage.span()),source);
            for(auto field:result_fields(s.result))disjoint(range(field),source);
            auto& out=s.outputs;
            for(auto field:{out.velocity,out.second_residual,out.spliced,out.unspliced,out.spliced_second,out.cross,out.unspliced_second})
                disjoint(range(field),source);
        }
    }
    if(applied)std::fill_n(applied,count,std::uint8_t(0));
    Guard guard(s.p->device);pack_base(s,first,count);
    if(original) {
        if(original->owner!=s.p)throw std::invalid_argument("alternate source belongs to another input");
        auto& genes=s.promoted_genes;auto& columns=s.promoted_columns;
        genes.clear();columns.clear();genes.reserve(count);columns.reserve(count);
        for(std::size_t j=0;j<count;++j)if(s.stochastic && !initialization
            ? s.promotions[first+j]==1 : (promotion && promotion[s.p->host_genes[first+j]])) {
            genes.push_back(s.p->host_genes[first+j]);columns.push_back(j);if(applied)applied[j]=2;
        }
        if(!genes.empty() && original->available) {
            auto x=original->spliced.view,y=original->unspliced.view;
            bool local=false;SourceBlock block;
            if(original->source.acquire) {
                original->source.acquire(original->source.context,genes.data(),genes.size(),s.stream,block);
                check(cudaSetDevice(s.p->device));
                bool any_available=!block.available;
                if(block.available)for(std::size_t k=0;k<genes.size();++k)any_available|=block.available[k]!=0;
                bool higher_precision=block.spliced.dtype!=ScalarType::float16 && block.unspliced.dtype!=ScalarType::float16;
                // An all-declined request needs no dummy numerical allocation.
                if(any_available && higher_precision) {
                    check_source_block(block,s.p->n,genes.size(),s.p->device);
                    check_source_aliases(s,block);
                }
                x={matrix(MatrixView(block.spliced))};y={matrix(MatrixView(block.unspliced))};local=true;
            }
            // O(block) maps only; source numerical values overwrite the
            // already allocated S/U columns before the one W application.
            for(std::size_t k=0;k<genes.size();++k) {
                bool half_source=local && (block.spliced.dtype==ScalarType::float16 || block.unspliced.dtype==ScalarType::float16);
                if(local && ((block.available && !block.available[k]) || half_source)) {
                    if(s.stochastic && !initialization)throw std::runtime_error("promoted original disappeared between stochastic passes");
                    continue;
                }
                auto source_gene=local?std::uint64_t(k):genes[k],destination=columns[k];
                check(bridge::pack_override(x,y,source_gene,destination,s.p->n,s.w->fields,
                    s.w->precision==ComputePrecision::float32,s.stream.native));
                if(applied)applied[columns[k]]=1;
                if(s.stochastic && initialization)s.promotions[first+columns[k]]=1;
            }
        }
        if(s.stochastic && initialization)for(std::size_t j=0;j<count;++j)
            if(promotion && promotion[s.p->host_genes[first+j]] && s.promotions[first+j]!=1)s.promotions[first+j]=2;
        if(s.stochastic && !initialization && applied)for(std::size_t j=0;j<count;++j)applied[j]=s.promotions[first+j];
    }
    bool second=s.stochastic && !initialization;
    check(bridge::moments(s.p->graph.view,s.p->n,count,s.w->fields,s.w->precision==ComputePrecision::float32,second,s.stream.native));
    return block_view(*s.w,first,count,second);
}
}
}

PreparedInput::PreparedInput(Input input,Stream stream,anndata::Validation validation) :state_(std::make_unique<detail::PreparedState>()) {
    auto& s=*state_;s.device=detail::device(input.weights);stream=detail::stream_for(stream,s.device);
    detail::validate_matrix(input.weights,s.device);s.n=detail::rows(input.weights);
    if(detail::columns(input.weights)!=s.n)throw std::invalid_argument("device weights must be square");
    if(input.source.acquire) {
        s.source_g=input.source_genes;
        if(!detail::floating(input.source_spliced_dtype)||!detail::floating(input.source_unspliced_dtype))throw std::invalid_argument("block source dtype must be floating");
    }
    else {
        detail::validate_matrix(input.spliced,s.device);detail::validate_matrix(input.unspliced,s.device);
        s.source_g=detail::columns(input.spliced);
        if(detail::rows(input.spliced)!=s.n || detail::rows(input.unspliced)!=s.n || detail::columns(input.unspliced)!=s.source_g)
            throw std::invalid_argument("device expression shapes differ");
    }
    s.g=input.genes.extent(s.source_g);s.host_genes.resize(s.g);
    for(std::size_t j=0;j<s.g;++j) {auto gene=input.genes.index(j);if(gene>=s.source_g)throw std::out_of_range("device selected gene exceeds source extent");s.host_genes[j]=gene;}
    if(input.fit_mask)detail::validate_mask(*input.fit_mask,s.n,s.device);
    if(input.gene_mask)detail::validate_mask(*input.gene_mask,s.source_g,s.device);
    s.checked=validation==anndata::Validation::full;
    if(s.checked) {
        detail::validate_values(input.weights,stream,true);
        if(!input.source.acquire) {detail::validate_values(input.spliced,stream,true);detail::validate_values(input.unspliced,stream,true);}
    }
    s.graph=detail::prepare_matrix(input.weights,true,stream);
    if(!input.source.acquire) {s.spliced=detail::prepare_matrix(input.spliced,false,stream);s.unspliced=detail::prepare_matrix(input.unspliced,false,stream);}
    s.genes=DeviceBuffer(detail::mul(s.g,sizeof(std::uint64_t)),s.device);
    if(s.g) {ac::copy_to_device(s.genes.span(),0,s.host_genes.data(),s.genes.size_bytes(),stream);ac::synchronize(stream);}
    // The retained host vector is stable through all asynchronous consumers.
    s.input=std::move(input);
}
PreparedInput::~PreparedInput()=default;
PreparedInput::PreparedInput(PreparedInput&&) noexcept=default;
PreparedInput& PreparedInput::operator=(PreparedInput&&) noexcept=default;
std::size_t PreparedInput::n_cells() const noexcept {return state_?state_->n:0;}
std::size_t PreparedInput::n_genes() const noexcept {return state_?state_->g:0;}
std::size_t PreparedInput::source_genes() const noexcept {return state_?state_->source_g:0;}
std::size_t PreparedInput::gene_index(std::size_t i) const {if(!state_ || i>=state_->g)throw std::out_of_range("device gene index");return state_->host_genes[i];}
std::size_t PreparedInput::storage_bytes() const noexcept {return state_?state_->graph.bytes()+state_->spliced.bytes()+state_->unspliced.bytes()+state_->genes.size_bytes()+state_->host_genes.capacity()*sizeof(std::uint64_t):0;}
int PreparedInput::device() const noexcept {return state_?state_->device:-1;}
void PreparedInput::validate_values(Stream stream) const {
    auto& s=detail::Access::get(*this);stream=detail::stream_for(stream,s.device);
    detail::validate_values(s.input.weights,stream,false);
    if(!s.input.source.acquire) {detail::validate_values(s.input.spliced,stream,false);detail::validate_values(s.input.unspliced,stream,false);}
}

ExpressionSource::ExpressionSource(const PreparedInput& p,MatrixView x,MatrixView y,Stream stream,anndata::Validation validation)
    :state_(std::make_unique<detail::ExpressionState>()) {
    auto& ps=detail::Access::get(p);auto& s=*state_;s.owner=&ps;stream=detail::stream_for(stream,ps.device);
    detail::validate_matrix(x,ps.device);detail::validate_matrix(y,ps.device);
    if(detail::rows(x)!=ps.n || detail::rows(y)!=ps.n || detail::columns(x)!=ps.source_g || detail::columns(y)!=ps.source_g)
        throw std::invalid_argument("alternate expression shapes differ");
    auto xt=x.sparse?x.compressed.dtype:x.dense.dtype,yt=y.sparse?y.compressed.dtype:y.dense.dtype;
    s.available=xt!=ScalarType::float16 && yt!=ScalarType::float16;
    if(s.available) {
        if(validation==anndata::Validation::full) {detail::validate_values(x,stream,true);detail::validate_values(y,stream,true);}
        s.spliced=detail::prepare_matrix(x,false,stream);s.unspliced=detail::prepare_matrix(y,false,stream);
    }
}
ExpressionSource::ExpressionSource(const PreparedInput& p,BlockSource source):state_(std::make_unique<detail::ExpressionState>()) {
    state_->owner=&detail::Access::get(p);state_->source=source;state_->available=source.acquire;
}
ExpressionSource::~ExpressionSource()=default;
ExpressionSource::ExpressionSource(ExpressionSource&&) noexcept=default;
ExpressionSource& ExpressionSource::operator=(ExpressionSource&&) noexcept=default;
std::size_t ExpressionSource::storage_bytes() const noexcept {return state_?state_->spliced.bytes()+state_->unspliced.bytes():0;}

Workspace::Workspace(const PreparedInput& p,RunOptions o):state_(detail::workspace(p,o.workspace_bytes,o.block_genes,o.compute_precision,false,false)) {}
Workspace::Workspace(const PreparedInput& p,StochasticRunOptions o):state_(detail::workspace(p,o.workspace_bytes,o.block_genes,o.compute_precision,true,o.unspliced_second)) {}
Workspace::~Workspace()=default;
Workspace::Workspace(Workspace&&) noexcept=default;
Workspace& Workspace::operator=(Workspace&&) noexcept=default;
std::size_t Workspace::block_genes() const noexcept {return state_?state_->block:0;}
std::size_t Workspace::scratch_bytes() const noexcept {return state_?state_->storage.size_bytes():0;}
std::size_t Workspace::staging_bytes() const noexcept {return state_?state_->staging:0;}
std::size_t Workspace::budget_bytes() const noexcept {return state_?state_->budget:0;}
ComputePrecision Workspace::compute_precision() const noexcept {return state_?state_->precision:ComputePrecision::float64;}

FitResult::FitResult(const PreparedInput& p,ComputePrecision precision,bool stochastic):state_(std::make_unique<detail::ResultState>()) {
    auto& ps=detail::Access::get(p);auto& s=*state_;auto type=detail::dtype(precision);auto width=ac::scalar_size(type);
    s.numeric=DeviceBuffer(detail::mul(detail::mul(ps.g,stochastic?9:3),width),ps.device);
    s.flags=DeviceBuffer(detail::add(detail::mul(ps.g,stochastic?3:2),2),ps.device);
    std::size_t offset=0;auto numeric=[&]() {auto v=detail::view(s.numeric.span(),offset,ps.g,1,type);offset+=ps.g*width;return v;};
    s.view.initial_gamma=numeric();s.view.initial_offset=numeric();s.view.r2=numeric();
    if(stochastic) {s.view.gamma=numeric();s.view.offset=numeric();s.view.offset2=numeric();s.view.first_scale=numeric();s.view.second_scale=numeric();s.view.objective=numeric();}
    else {s.view.gamma=s.view.initial_gamma;s.view.offset=s.view.initial_offset;}
    offset=0;auto flags=[&](std::size_t n) {auto v=detail::view(s.flags.span(),offset,n,1,ScalarType::uint8);offset+=n;return v;};
    s.view.selected=flags(ps.g);s.view.initial_status=flags(ps.g);s.view.status=stochastic?flags(ps.g):s.view.initial_status;
    s.view.low_signal=flags(1);s.view.refit_all=flags(1);
}
FitResult::~FitResult()=default;
FitResult::FitResult(FitResult&&) noexcept=default;
FitResult& FitResult::operator=(FitResult&&) noexcept=default;
ResultView FitResult::view() const noexcept {return state_?state_->view:ResultView{};}
std::size_t FitResult::storage_bytes() const noexcept {return state_?state_->numeric.size_bytes()+state_->flags.size_bytes():0;}

ReferenceStages::ReferenceStages(const PreparedInput& p,Workspace& w,ResultView r,FitOptions o,Outputs out,Stream stream,BlockSink sink)
    :state_(detail::stages(p,w,r,StochasticOptions{o},out,stream,false,sink)) {}
ReferenceStages::ReferenceStages(const PreparedInput& p,Workspace& w,ResultView r,StochasticOptions o,Outputs out,Stream stream,BlockSink sink)
    :state_(detail::stages(p,w,r,o,out,stream,true,sink)) {}
ReferenceStages::~ReferenceStages()=default;
ReferenceStages::ReferenceStages(ReferenceStages&&) noexcept=default;
ReferenceStages& ReferenceStages::operator=(ReferenceStages&&) noexcept=default;
MomentBlock ReferenceStages::initial_moments(std::size_t first,std::size_t count) {return detail::calculate(*state_,first,count,true);}
MomentBlock ReferenceStages::initial_moments_from(const ExpressionSource& source,std::size_t first,std::size_t count,
    const std::uint8_t* mask,std::uint8_t* applied) {return detail::calculate(*state_,first,count,true,&detail::Access::get(source),mask,applied);}
MomentBlock ReferenceStages::moments(std::size_t first,std::size_t count) {return detail::calculate(*state_,first,count,false);}
MomentBlock ReferenceStages::moments_from(const ExpressionSource& source,std::size_t first,std::size_t count,
    const std::uint8_t* mask,std::uint8_t* applied) {return detail::calculate(*state_,first,count,false,&detail::Access::get(source),mask,applied);}
FitBlock ReferenceStages::initial_fit(MomentBlock m) {
    auto& s=*state_;auto f=detail::moment_fields(s,m,false);detail::Guard guard(s.p->device);
    detail::check(bridge::initial_fit(s.p->n,m.first,m.count,static_cast<const std::uint64_t*>(s.p->genes.data()),
        s.p->input.fit_mask.value_or(DenseView{}),s.p->input.gene_mask.value_or(DenseView{}),f,s.raw,s.options,
        s.w->precision==ComputePrecision::float32,s.stream.native));return {m.first,m.count,s.result};
}
void ReferenceStages::accept_initialization(MomentBlock m,FitBlock f) {
    auto& s=*state_;detail::check_range(s,m.first,m.count,true);detail::check_fit_block(s,m,f);s.next+=m.count;
}
void ReferenceStages::resolve_selection() {
    auto& s=*state_;
    if(!s.stochastic || s.resolved || s.finished || s.next!=s.p->g)throw std::invalid_argument("unfinished CUDA initialization");
    detail::Guard guard(s.p->device);detail::check(bridge::resolve(s.p->g,static_cast<const std::uint64_t*>(s.p->genes.data()),
        s.p->input.gene_mask.value_or(DenseView{}),s.raw,s.options,s.w->fields.scores,s.w->precision==ComputePrecision::float32,true,s.stream.native));
    s.resolved=true;s.next=0;
}
FitBlock ReferenceStages::scales(MomentBlock m) {
    auto& s=*state_;if(!s.stochastic || !s.resolved)throw std::invalid_argument("CUDA scales require resolved stochastic stages");
    auto f=detail::moment_fields(s,m,true);detail::Guard guard(s.p->device);
    detail::check(bridge::scales(s.p->n,m.first,m.count,static_cast<const std::uint64_t*>(s.p->genes.data()),
        s.p->input.fit_mask.value_or(DenseView{}),s.p->input.gene_mask.value_or(DenseView{}),f,s.raw,s.options,
        s.w->precision==ComputePrecision::float32,s.stream.native));return {m.first,m.count,s.result};
}
FitBlock ReferenceStages::fit(MomentBlock m) {
    if(state_->stochastic)return fit(m,scales(m));
    return initial_fit(m);
}
FitBlock ReferenceStages::fit(MomentBlock m,FitBlock scales) {
    auto& s=*state_;if(!s.stochastic || !s.resolved)throw std::invalid_argument("CUDA joint fit requires resolved stochastic stages");
    detail::check_fit_block(s,m,scales);auto f=detail::moment_fields(s,m,true);detail::Guard guard(s.p->device);
    detail::check(bridge::fit(s.p->n,m.first,m.count,static_cast<const std::uint64_t*>(s.p->genes.data()),
        s.p->input.fit_mask.value_or(DenseView{}),s.p->input.gene_mask.value_or(DenseView{}),f,s.raw,s.options,
        s.w->precision==ComputePrecision::float32,s.stream.native));return {m.first,m.count,s.result};
}
void ReferenceStages::emit(MomentBlock m,FitBlock fit) {
    auto& s=*state_;detail::check_fit_block(s,m,fit);auto f=detail::moment_fields(s,m,s.stochastic);detail::Guard guard(s.p->device);
    auto& o=s.outputs;detail::check(bridge::emit(s.p->n,m.first,m.count,f,s.raw,
        {o.velocity,o.second_residual,o.spliced,o.unspliced,o.spliced_second,o.cross,o.unspliced_second,o.block_local?1u:0u},
        s.w->precision==ComputePrecision::float32,s.stochastic,s.stream.native));
    if(s.sink.write) {
        auto output=o;
        auto slice=[&](DenseView v) {
            if(v.values.data) {
                if(!o.block_local)v.origin=std::uint64_t(__int128(v.origin)+__int128(m.first)*v.column_stride*ac::scalar_size(v.dtype));
                v.columns=m.count;
            }
            return v;
        };
        output.velocity=slice(o.velocity);output.second_residual=slice(o.second_residual);
        output.spliced=slice(o.spliced);output.unspliced=slice(o.unspliced);
        output.spliced_second=slice(o.spliced_second);output.cross=slice(o.cross);output.unspliced_second=slice(o.unspliced_second);
        s.sink.write(s.sink.context,OutputBlock{m.first,m.count,output},s.stream);
    }
    s.next+=m.count;
}
void ReferenceStages::sensitivity(MomentBlock m,FitBlock f,SensitivityView v,velocity::precision::SensitivityOptions options) {
    auto& s=*state_;
    if(s.stochastic && s.resolved)throw std::invalid_argument("sensitivity requires pre-global initialization fit");
    detail::check_fit_block(s,m,f);auto fields=detail::moment_fields(s,m,false);
    if(!std::isfinite(options.band)||options.band<0||!std::isfinite(options.prediction_limit)||options.prediction_limit<0
        ||!std::isfinite(options.r2_limit)||options.r2_limit<0)throw std::invalid_argument("invalid sensitivity options");
    auto check_output=[&](DenseView output,ScalarType type,bool required) {
        if(!required && !output.values.data)return;
        ac::validate(output,true);
        if(output.rows!=s.p->g || output.columns!=1 || output.dtype!=type || output.values.device!=s.p->device
            ||(output.rows>1 && !output.row_stride))throw std::invalid_argument("sensitivity view shape/type/stride mismatch");
        auto target=detail::range(output);
        if(detail::overlaps(target,detail::range(s.w->storage.span())))throw std::invalid_argument("sensitivity output aliases workspace");
        for(auto field:detail::result_fields(s.result))if(detail::overlaps(target,detail::range(field)))throw std::invalid_argument("sensitivity output aliases fit");
        for(auto field:{m.spliced,m.unspliced})if(detail::overlaps(target,detail::range(field)))throw std::invalid_argument("sensitivity output aliases moments");
        if((s.p->input.fit_mask && detail::overlaps(target,detail::range(*s.p->input.fit_mask)))
           ||(s.p->input.gene_mask && detail::overlaps(target,detail::range(*s.p->input.gene_mask))))
            throw std::invalid_argument("sensitivity output aliases mask");
        detail::disjoint(target,s.p->input.weights);
        if(!s.p->input.source.acquire) {detail::disjoint(target,s.p->input.spliced);detail::disjoint(target,s.p->input.unspliced);}
    };
    check_output(v.state,ScalarType::uint8,true);
    for(auto output:{v.low,v.high,v.delta,v.prediction_change,v.r2_change})check_output(output,detail::dtype(s.w->precision),false);
    for(auto output:{v.expanded_status,v.contracted_status,v.status_change,v.selection_change})check_output(output,ScalarType::uint8,false);
    std::array<DenseView,10> reports{v.state,v.low,v.high,v.delta,v.prediction_change,v.r2_change,
        v.expanded_status,v.contracted_status,v.status_change,v.selection_change};
    for(std::size_t i=0;i<reports.size();++i)for(std::size_t j=0;j<i;++j)
        if(detail::overlaps(detail::range(reports[i]),detail::range(reports[j])))throw std::invalid_argument("sensitivity outputs overlap");
    detail::Guard guard(s.p->device);detail::check(bridge::sensitivity(s.p->n,m.first,m.count,
        static_cast<const std::uint64_t*>(s.p->genes.data()),s.p->input.fit_mask.value_or(DenseView{}),
        s.p->input.gene_mask.value_or(DenseView{}),fields,s.raw,s.options,
        {options.band,options.prediction_limit,options.r2_limit},
        {v.state,v.low,v.high,v.delta,v.prediction_change,v.r2_change,v.expanded_status,v.contracted_status,v.status_change,v.selection_change},
        s.w->precision==ComputePrecision::float32,s.stream.native));
}
ResultView ReferenceStages::finish() {
    auto& s=*state_;if(s.finished || !s.resolved || s.next!=s.p->g)throw std::invalid_argument("unfinished CUDA blocks");
    if(!s.stochastic) {detail::Guard guard(s.p->device);detail::check(bridge::resolve(s.p->g,
        static_cast<const std::uint64_t*>(s.p->genes.data()),s.p->input.gene_mask.value_or(DenseView{}),s.raw,s.options,
        s.w->fields.scores,s.w->precision==ComputePrecision::float32,false,s.stream.native));}
    s.finished=true;return s.result;
}

namespace reference {
void steady_state_into(const PreparedInput& p,Workspace& w,ResultView result,FitOptions o,Outputs outputs,Stream stream,BlockSink sink) {
    ReferenceStages stages(p,w,result,o,outputs,stream,sink);velocity::steady_state_with(stages,p.n_genes(),w.block_genes());
}
void stochastic_into(const PreparedInput& p,Workspace& w,ResultView result,StochasticOptions o,Outputs outputs,Stream stream,BlockSink sink) {
    ReferenceStages stages(p,w,result,o,outputs,stream,sink);velocity::stochastic_with(stages,p.n_genes(),w.block_genes());
}
FitResult steady_state(const PreparedInput& p,Workspace& w,FitOptions o,Outputs outputs,Stream stream,BlockSink sink) {
    FitResult result(p,w.compute_precision());steady_state_into(p,w,result.view(),o,outputs,stream,sink);return result;
}
FitResult stochastic(const PreparedInput& p,Workspace& w,StochasticOptions o,Outputs outputs,Stream stream,BlockSink sink) {
    FitResult result(p,w.compute_precision(),true);stochastic_into(p,w,result.view(),o,outputs,stream,sink);return result;
}
void prepare_neighborhood(MatrixView input,DenseView output,Stream stream) {
    auto device=detail::device(input);stream=detail::stream_for(stream,device);detail::validate_matrix(input,device);ac::validate(output,true);
    if(detail::rows(input)!=detail::columns(input) || output.rows!=detail::rows(input) || output.columns!=detail::columns(input)
       ||(output.dtype!=ScalarType::float32 && output.dtype!=ScalarType::float64) ||output.values.device!=device)
        throw std::invalid_argument("neighborhood output must be matching dense FP32/FP64");
    if(!detail::separate_elements(output))throw std::invalid_argument("neighborhood output logical elements overlap");
    detail::disjoint(detail::range(output),input);detail::validate_values(input,stream,true);
    auto prepared=detail::prepare_matrix(input,true,stream);detail::Guard guard(device);
    detail::check(bridge::neighborhood(prepared.view,output,stream.native));
    // Temporary row index is local to this explicit preparation helper.
    ac::synchronize(stream);
}
}
}
