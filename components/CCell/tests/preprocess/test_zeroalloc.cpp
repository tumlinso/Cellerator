// Allocation instrumentation is glibc-specific and observes public malloc-family
// symbols plus C++ new/new[]. It cannot see private allocators or direct mmap.
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <Cellerator/memory/view.hh>
#include <Cellerator/tensor/base/compressed.hh>

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define CCELL_NO_ALLOCATION_INTERPOSITION 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) && !defined(CCELL_NO_ALLOCATION_INTERPOSITION)
#define CCELL_NO_ALLOCATION_INTERPOSITION 1
#endif

#if defined(__linux__) && defined(__GLIBC__)
import preprocess;

#if !defined(CCELL_NO_ALLOCATION_INTERPOSITION)
static std::atomic<bool> measuring{false};
static std::atomic<size_t> c_calls{0}, cpp_calls{0};
extern "C" void* __libc_malloc(size_t);
extern "C" void* __libc_calloc(size_t, size_t);
extern "C" void* __libc_realloc(void*, size_t);
extern "C" void __libc_free(void*);
extern "C" void* __libc_memalign(size_t, size_t);
static void count_c() { if (measuring.load(std::memory_order_relaxed)) ++c_calls; }
static void count_cpp() { if (measuring.load(std::memory_order_relaxed)) ++cpp_calls; }
extern "C" void* malloc(size_t n) noexcept { count_c(); return __libc_malloc(n); }
extern "C" void* calloc(size_t n, size_t s) noexcept { count_c(); return __libc_calloc(n,s); }
extern "C" void* realloc(void* p, size_t n) noexcept { count_c(); return __libc_realloc(p,n); }
extern "C" void free(void* p) noexcept { __libc_free(p); }
extern "C" void* aligned_alloc(size_t a, size_t n) noexcept { count_c(); return __libc_memalign(a,n); }
extern "C" int posix_memalign(void** p, size_t a, size_t n) noexcept {
    count_c(); if (a < sizeof(void*) || (a & (a-1)) || a % sizeof(void*)) return 22;
    void* q = __libc_memalign(a,n); if (!q) return 12; *p = q; return 0;
}
void* operator new(size_t n) { count_cpp(); if (void* p = malloc(n)) return p; throw std::bad_alloc(); }
void* operator new[](size_t n) { count_cpp(); if (void* p = malloc(n)) return p; throw std::bad_alloc(); }
void* operator new(size_t n, std::align_val_t a) { count_cpp(); count_c(); if (void* p = __libc_memalign(size_t(a),n)) return p; throw std::bad_alloc(); }
void* operator new[](size_t n, std::align_val_t a) { count_cpp(); count_c(); if (void* p = __libc_memalign(size_t(a),n)) return p; throw std::bad_alloc(); }
void operator delete(void* p) noexcept { free(p); }
void operator delete[](void* p) noexcept { free(p); }
void operator delete(void* p, size_t) noexcept { free(p); }
void operator delete[](void* p, size_t) noexcept { free(p); }
void operator delete(void* p, std::align_val_t) noexcept { free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { free(p); }
void operator delete(void* p, size_t, std::align_val_t) noexcept { free(p); }
void operator delete[](void* p, size_t, std::align_val_t) noexcept { free(p); }
#endif

static int failures = 0;
static void add_block(void* context, size_t first, size_t count) {
    auto* total = static_cast<size_t*>(context); *total += first + count;
}
template<class F> auto measure(const char* name, F f) {
#if defined(CCELL_NO_ALLOCATION_INTERPOSITION)
    (void)name;
    return f();
#else
    auto before_c = c_calls.load(), before_cpp = cpp_calls.load();
    measuring = true; auto result = f(); measuring = false;
    auto dc = c_calls.load()-before_c, dn = cpp_calls.load()-before_cpp;
    if (dc || dn) { std::fprintf(stderr, "%s allocated: C=%zu C++=%zu\n", name, dc, dn); ++failures; }
    return result;
#endif
}
template<class F> void measure_void(const char* name, F f) {
    measure(name, [&] { f(); return 0; });
}
static void expect(bool ok, const char* name) { if (!ok) { std::fprintf(stderr,"unexpected result: %s\n",name); ++failures; } }

int main() {
    namespace p = preprocess;
    using cellerator::memory::const_matrix_view;
    using cellerator::memory::matrix_view;
    using cellerator::tensor::base::CompressedAxis;
    using cellerator::tensor::base::CompressedView;
#if defined(CCELL_NO_ALLOCATION_INTERPOSITION)
    std::fprintf(stderr, "Allocation measurement disabled under AddressSanitizer; exercising numerical behavior only.\n");
#else
    // Verify the instrumentation before measuring the numerical calls.
    measuring = true;
    void* (*volatile m)(size_t) = &malloc;
    void* (*volatile c)(size_t,size_t) = &calloc;
    void* (*volatile r)(void*,size_t) = &realloc;
    void* sanity_c = m(17); void* sanity_cr = c(3,8); sanity_cr = r(sanity_cr,64);
    void* sanity_new = ::operator new(19); void* sanity_array = ::operator new[](23);
    void* sanity_aligned = ::operator new(64,std::align_val_t(32));
    measuring = false;
    free(sanity_c); free(sanity_cr); ::operator delete(sanity_new); ::operator delete[](sanity_array);
    ::operator delete(sanity_aligned,std::align_val_t(32));
    expect(c_calls.load() >= 6 && cpp_calls.load() >= 3, "allocator interception sanity");
#endif

    double values[16], result[16];
    for (size_t i=0;i<16;++i) values[i] = .1*double((i*7)%13+1);
    const_matrix_view<double> dense{values,4,4,4};
    matrix_view<double> output{result,4,4,4};
    int32_t indices[16];
    int64_t offsets[5];
    for (size_t i=0;i<4;++i) {
        offsets[i]=int64_t(4*i);
        for (size_t j=0;j<4;++j) indices[4*i+j]=int32_t(j);
    }
    offsets[4]=16;
    CompressedView<int32_t,const double,int64_t> sparse{{4,4,16,CompressedAxis::row,offsets,indices},values};
    CompressedView<int32_t,double,int64_t> sparse_out{sparse.structure,result};
    const_matrix_view<double> empty;
    matrix_view<double> empty_out;

    double ct[4]{}, gt[4]{}, mean[4]{}, drop[4]{}, lct[4]{}, lcd[4]{}, lgt[4]{}, lgm[4]{};
    double groups[4]{}, group_pct[4]{}, group_log[4]{}, tops[8]{}, scratch[4]{}, var[4]{};
    uint64_t cd[4]{}, gd[4]{}, counts[4]{}; uint8_t mask[4]{1,0,1,0}, keep[4]{};
    size_t topn[2]{1,3};
    p::QcOptions qo{mask,1,topn,2}; p::QcBuffers qb;
    qb.cell_total=ct; qb.cell_detected=cd; qb.cells_capacity=4;
    qb.gene_total=gt; qb.gene_detected=gd; qb.genes_capacity=4; qb.mean=mean; qb.dropout=drop;
    qb.log_cell_total=lct; qb.log_cell_detected=lcd; qb.log_gene_total=lgt; qb.log_gene_mean=lgm;
    qb.group_total=groups; qb.group_percent=group_pct; qb.group_log_total=group_log; qb.group_capacity=4;
    qb.top_percent=tops; qb.top_capacity=8;
    expect(bool(measure("qc check",[&]{return p::qc_check(4,4,qo,qb,4); })),"qc check");
    expect(bool(measure("qc validate dense",[&]{return p::qc_validate(dense,true);})),"qc validate dense");
    expect(bool(measure("qc validate csr",[&]{return p::qc_validate(sparse,true);})),"qc validate csr");
    expect(!bool(measure("qc capacity",[&]{return p::qc_check(4,4,qo,qb,0); })),"qc capacity");
    measure_void("qc zero",[&]{p::qc_zero(4,4,1,qb);});
    ct[0] = 42; scratch[0] = 47;
    auto qc_short = measure("qc partial dense capacity",[&]{return p::qc_accumulate(dense,0,4,qo,qb,scratch,0);});
    expect(qc_short.code == p::QcCode::insufficient_capacity && ct[0] == 42 && scratch[0] == 47,"qc partial dense capacity preflight");
    qc_short = measure("qc partial csr capacity",[&]{return p::qc_accumulate(sparse,0,4,qo,qb,scratch,0);});
    expect(qc_short.code == p::QcCode::insufficient_capacity && ct[0] == 42 && scratch[0] == 47,"qc partial csr capacity preflight");
    measure_void("qc zero after short",[&]{p::qc_zero(4,4,1,qb);});
    expect(bool(measure("qc partial dense",[&]{return p::qc_accumulate(dense,0,4,qo,qb,scratch,4);})),"qc partial dense");
    measure_void("qc zero sparse",[&]{p::qc_zero(4,4,1,qb);});
    expect(bool(measure("qc partial csr",[&]{return p::qc_accumulate(sparse,0,4,qo,qb,scratch,4);})),"qc partial csr");
    measure_void("qc zero unchecked dense",[&]{p::qc_zero(4,4,1,qb);});
    expect(bool(measure("qc unchecked dense",[&]{return p::qc_accumulate_unchecked(dense,0,4,qo,qb,scratch);})),"qc unchecked dense");
    measure_void("qc zero unchecked csr",[&]{p::qc_zero(4,4,1,qb);});
    expect(bool(measure("qc unchecked csr",[&]{return p::qc_accumulate_unchecked(sparse,0,4,qo,qb,scratch);})),"qc unchecked csr");
    measure_void("qc finalize",[&]{p::qc_finish(4,4,qo,qb);});
    double mt[4]{}; uint64_t md[4]{};
    measure_void("qc merge",[&]{p::qc_merge(4,mt,md,gt,gd);});
    for (int i=0;i<2;++i) {
        expect(bool(measure("qc dense",[&]{return p::qc(dense,qo,qb,scratch,4);})),"qc dense");
        expect(bool(measure("qc csr",[&]{return p::qc(sparse,qo,qb,scratch,4);})),"qc csr");
    }
    auto invalid_dense=dense; invalid_dense.data=nullptr;
    auto qs=measure("qc missing data",[&]{return p::qc(invalid_dense,qo,qb,scratch,4);});
    expect(qs.code==p::QcCode::invalid_argument,"qc missing data");
    const long double unsupported_values[16]={};
    const_matrix_view<long double> unsupported{unsupported_values,4,4,4};
    qs=measure("qc unsupported dtype",[&]{return p::qc(unsupported,qo,qb,scratch,4);});
    expect(qs.code==p::QcCode::unsupported_type,"qc unsupported dtype");
    p::QcOptions empty_options; p::QcBuffers empty_outputs;
    expect(bool(measure("qc empty",[&]{return p::qc(empty,empty_options,empty_outputs,nullptr,0);})),"qc empty");
    p::FilterBounds fb; fb.has_min_counts=true; fb.min_counts=0;
    expect(bool(measure("filter",[&]{return p::filter(ct,cd,4,fb,keep,4);})),"filter");
    expect(!bool(measure("filter capacity",[&]{return p::filter(ct,cd,4,fb,keep,0);})),"filter capacity");
    size_t observed=0;
    expect(bool(measure("stats dense",[&]{return p::feature_stats(dense,nullptr,mean,var,4,counts,4,&observed);})),"stats dense");
    expect(bool(measure("stats csr",[&]{return p::feature_stats(sparse,nullptr,mean,var,4,counts,4,&observed);})),"stats csr");
    expect(bool(measure("stats dense unchecked",[&]{return p::feature_stats_unchecked(dense,nullptr,mean,var,4,counts,4,&observed);})),"stats dense unchecked");
    expect(bool(measure("stats csr unchecked",[&]{return p::feature_stats_unchecked(sparse,nullptr,mean,var,4,counts,4,&observed);})),"stats csr unchecked");
    double pm1[4]{}, pm2[4]{}, pv1[4]{}, pv2[4]{};
    p::StatsPartial partial1{0,4,pm1,pv1,4}, partial2{0,4,pm2,pv2,4};
    expect(bool(measure("stats partial dense",[&]{return p::stats_partial(dense,0,2,nullptr,partial1,counts,4);})),"stats partial dense");
    expect(bool(measure("stats partial csr",[&]{return p::stats_partial(sparse,2,4,nullptr,partial2,counts,4);})),"stats partial csr");
    expect(bool(measure("stats merge",[&]{return p::stats_merge(partial1,partial2);})),"stats merge");
    expect(bool(measure("stats finish",[&]{return p::stats_finish(partial1,var,4);})),"stats finish");
    expect(!bool(measure("stats capacity",[&]{return p::feature_stats(dense,nullptr,mean,var,0,counts,4,&observed);})),"stats capacity");
    expect(bool(measure("stats empty",[&]{return p::feature_stats(empty,nullptr,nullptr,nullptr,0,nullptr,0,&observed);})) && observed==0,"stats empty");

    double factors[4]{}, nwork[4]{}, scales[4]{}; uint8_t included[4]{};
    p::NormalizationOptions no; no.target_is_set=true; no.target_sum=10;
    p::NormalizationBuffers nb{factors,4,included,4,nwork,4,1};
    expect(bool(measure("normalization prepare dense",[&]{return p::normalization_factors(dense,no,nb);})),"normalization prepare dense");
    expect(bool(measure("normalization prepare csr",[&]{return p::normalization_factors(sparse,no,nb);})),"normalization prepare csr");
    auto automatic=no; automatic.target_is_set=false;
    expect(bool(measure("normalization automatic target",[&]{return p::normalization_factors(dense,automatic,nb);})),"normalization automatic target");
    auto short_nb=nb; short_nb.scratch_capacity=0;
    expect(!bool(measure("normalization capacity",[&]{return p::normalization_factors(dense,automatic,short_nb);})),"normalization capacity");
    p::NormalizationBuffers empty_nb;
    expect(bool(measure("normalization prepare empty",[&]{return p::normalization_factors(empty,automatic,empty_nb);})),"normalization prepare empty");
    expect(bool(measure("normalize empty",[&]{return p::normalize_into(empty,empty_out,automatic,empty_nb);})),"normalize empty");
    expect(bool(measure("log empty",[&]{return p::log1p_into(empty,empty_out);})),"log empty");
    for (int i=0;i<2;++i) {
        expect(bool(measure("normalize dense",[&]{return p::normalize_into(dense,output,no,nb);})),"normalize dense");
        expect(bool(measure("normalize csr to dense",[&]{return p::normalize_into(sparse,output,no,nb);})),"normalize csr to dense");
        expect(bool(measure("normalize csr",[&]{return p::normalize_into(sparse,sparse_out,no,nb);})),"normalize csr");
        expect(bool(measure("log dense",[&]{return p::log1p_into(dense,output);})),"log dense");
        expect(bool(measure("log csr to dense",[&]{return p::log1p_into(sparse,output);})),"log csr to dense");
        expect(bool(measure("log csr",[&]{return p::log1p_into(sparse,sparse_out);})),"log csr");
    }
    auto trusted_no=no; trusted_no.trusted=true;
    expect(bool(measure("trusted normalization prepare",[&]{return p::normalization_factors(dense,trusted_no,nb);})),"trusted normalization prepare");
    expect(bool(measure("trusted normalize",[&]{return p::normalize_into(dense,output,trusted_no,nb);})),"trusted normalize");
    expect(bool(measure("apply normalization",[&]{return p::apply_normalization(dense,output,factors,4);})),"apply normalization");
    expect(bool(measure("trusted apply normalization",[&]{return p::apply_normalization(dense,output,factors,4,true);})),"trusted apply normalization");
    expect(bool(measure("apply normalization csr dense",[&]{return p::apply_normalization(sparse,output,factors,4);})),"apply normalization csr dense");
    expect(bool(measure("apply normalization csr",[&]{return p::apply_normalization(sparse,sparse_out,factors,4);})),"apply normalization csr");
    expect(bool(measure("trusted log",[&]{return p::log1p_into(dense,output,0,true);})),"trusted log");
    double prior = result[0];
    auto transposed_output=sparse_out;
    transposed_output.structure.axis=CompressedAxis::column;
    auto wrong = measure("wrong sparse orientation",[&]{return p::normalize_into(sparse,transposed_output,no,nb);});
    expect(!bool(wrong) && result[0] == prior,"wrong sparse orientation preflight");
    expect(!bool(measure("log invalid base",[&]{return p::log1p_into(dense,output,1);})),"log invalid base");
    p::ScaleOptions sco;
    expect(bool(measure("scale empty",[&]{return p::scale_into(empty,empty_out,sco,nullptr,nullptr,0,nullptr,0);})),"scale empty");
    expect(bool(measure("scale dense",[&]{return p::scale_into(dense,output,sco,mean,scales,4,counts,4);})),"scale dense");
    expect(bool(measure("scale csr dense",[&]{return p::scale_into(sparse,output,sco,mean,scales,4,counts,4);})),"scale csr dense");
    sco.zero_center=false;
    expect(bool(measure("scale csr",[&]{return p::scale_into(sparse,sparse_out,sco,mean,scales,4,counts,4);})),"scale csr");
    expect(bool(measure("apply scale",[&]{return p::apply_scale(dense,output,sco,mean,scales,4);})),"apply scale");
    expect(bool(measure("apply scale csr dense",[&]{return p::apply_scale(sparse,output,sco,mean,scales,4);})),"apply scale csr dense");
    expect(bool(measure("apply scale csr",[&]{return p::apply_scale(sparse,sparse_out,sco,mean,scales,4);})),"apply scale csr");
    auto trusted_scale=sco; trusted_scale.trusted=true;
    expect(bool(measure("trusted scale",[&]{return p::scale_into(dense,output,trusted_scale,mean,scales,4,counts,4);})),"trusted scale");
    expect(!bool(measure("scale capacity",[&]{return p::scale_into(dense,output,sco,mean,scales,0,counts,4);})),"scale capacity");

    size_t total = 0;
    auto custom = [&](size_t first, size_t count) { total += first + count; };
    expect(bool(measure("static binding",[&]{return p::for_blocks(4,2,custom);})),"static binding");
    expect(bool(measure("runtime binding",[&]{return p::for_blocks(4,2,add_block,&total);})),"runtime binding");
    expect(bool(measure("empty binding",[&]{return p::for_blocks(0,2,add_block,&total);})),"empty binding");
    expect(!bool(measure("invalid binding",[&]{return p::for_blocks(4,0,add_block,&total);})),"invalid binding");

    p::HvgOptions ho; ho.n_bins=2;
    double hm[4]{}, hd[4]{}, hn[4]{}; uint8_t hs[4]{}, hi[4]{}; size_t hb[4]{};
    p::HvgOutput hout{hm,hd,hn,hs,hi,hb,4};
    alignas(std::max_align_t) unsigned char work[65536]; size_t needed=0;
    double sum1[4]{}, square1[4]{}, sum2[4]{}, square2[4]{};
    uint8_t expressed1[4]{}, expressed2[4]{}; size_t count1[1]{}, count2[1]{};
    p::HvgMoments moments1{sum1,square1,expressed1,count1,4,1};
    p::HvgMoments moments2{sum2,square2,expressed2,count2,4,1};
    auto mh=measure("hvg moments clear",[&]{return p::hvg_moments_clear(4,1,moments1);}); expect(mh.code==p::HvgCode::ok,"hvg moments clear");
    mh=measure("hvg moments clear 2",[&]{return p::hvg_moments_clear(4,1,moments2);}); expect(mh.code==p::HvgCode::ok,"hvg moments clear 2");
    auto short_moments=moments1; short_moments.capacity=0;
    mh=measure("hvg moments capacity",[&]{return p::hvg_moments_clear(4,1,short_moments);}); expect(mh.code!=p::HvgCode::ok,"hvg moments capacity");
    mh=measure("hvg accumulate dense",[&]{return p::hvg_accumulate_dense(dense,nullptr,1,ho,moments1);}); expect(mh.code==p::HvgCode::ok,"hvg accumulate dense");
    mh=measure("hvg accumulate csr",[&]{return p::hvg_accumulate_sparse(sparse,nullptr,1,ho,moments2);}); expect(mh.code==p::HvgCode::ok,"hvg accumulate csr");
    mh=measure("hvg merge",[&]{return p::hvg_moments_merge(4,1,moments1,moments2);}); expect(mh.code==p::HvgCode::ok,"hvg merge");
    mh=measure("hvg score size",[&]{return p::hvg_score_scratch_bytes(4,ho,needed);}); expect(mh.code==p::HvgCode::ok && needed<=sizeof(work),"hvg score size");
    mh=measure("hvg score capacity",[&]{return p::hvg_score(moments1,4,1,ho,hout,work,0);}); expect(mh.code!=p::HvgCode::ok,"hvg score capacity");
    for (int i=0;i<2;++i) {
        mh=measure("hvg score",[&]{return p::hvg_score(moments1,4,1,ho,hout,work,sizeof(work));}); expect(mh.code==p::HvgCode::ok,"hvg score");
        mh=measure("hvg score unchecked",[&]{return p::hvg_score_unchecked(moments1,4,1,ho,hout,work,sizeof(work));}); expect(mh.code==p::HvgCode::ok,"hvg score unchecked");
        mh=measure("hvg accumulate dense unchecked",[&]{return p::hvg_accumulate_dense_unchecked(dense,nullptr,1,ho,moments1);}); expect(mh.code==p::HvgCode::ok,"hvg accumulate dense unchecked");
        mh=measure("hvg accumulate csr unchecked",[&]{return p::hvg_accumulate_sparse_unchecked(sparse,nullptr,1,ho,moments2);}); expect(mh.code==p::HvgCode::ok,"hvg accumulate csr unchecked");
    }
    auto hq=measure("hvg size",[&]{return p::hvg_scratch_bytes(4,1,ho,needed);});
    expect(hq.code==p::HvgCode::ok && needed<=sizeof(work),"hvg size");
    measure("hvg empty size",[&]{return p::hvg_scratch_bytes(0,0,ho,needed);});
    hq=measure("hvg size repeated",[&]{return p::hvg_scratch_bytes(4,1,ho,needed);});
    expect(hq.code==p::HvgCode::ok,"hvg size repeated");
    for (int i=0;i<2;++i) {
        auto h=measure("hvg dense",[&]{return p::hvg_dense(dense,nullptr,1,ho,hout,work,sizeof(work));});
        expect(h.code==p::HvgCode::ok,"hvg dense");
        h=measure("hvg csr",[&]{return p::hvg_sparse(sparse,nullptr,1,ho,hout,work,sizeof(work));});
        expect(h.code==p::HvgCode::ok,"hvg csr");
        h=measure("hvg dense unchecked",[&]{return p::hvg_dense_unchecked(dense,nullptr,1,ho,hout,work,sizeof(work));});
        expect(h.code==p::HvgCode::ok,"hvg dense unchecked");
        h=measure("hvg csr unchecked",[&]{return p::hvg_sparse_unchecked(sparse,nullptr,1,ho,hout,work,sizeof(work));});
        expect(h.code==p::HvgCode::ok,"hvg csr unchecked");
    }
    auto h=measure("hvg capacity",[&]{return p::hvg_dense(dense,nullptr,1,ho,hout,work,0);});
    expect(h.code!=p::HvgCode::ok,"hvg capacity");
    size_t batch[4]{0,1,0,1}; auto batched=ho; batched.batched=true;
    h=measure("hvg batched size",[&]{return p::hvg_scratch_bytes(4,2,batched,needed);}); expect(h.code==p::HvgCode::ok && needed<=sizeof(work),"hvg batched size");
    h=measure("hvg batched",[&]{return p::hvg_dense(dense,batch,2,batched,hout,work,sizeof(work));}); expect(h.code==p::HvgCode::ok,"hvg batched");
    size_t order[4]{};
    for (int i=0;i<2;++i) {
        h=measure("hvg rank",[&]{return p::hvg_rank_batched(hn,hb,4,2,hs,order,4);}); expect(h.code==p::HvgCode::ok,"hvg rank");
    }
    h=measure("hvg rank capacity",[&]{return p::hvg_rank_batched(hn,hb,4,2,hs,order,0);}); expect(h.code==p::HvgCode::insufficient_capacity,"hvg rank capacity");
    measure("hvg rank empty",[&]{return p::hvg_rank_batched(nullptr,nullptr,0,0,nullptr,nullptr,0);});
    auto invalid_sparse=sparse;
    invalid_sparse.structure.axis=static_cast<CompressedAxis>(255);
    auto bad_layout=measure("hvg sparse invalid layout",[&]{return p::hvg_sparse(invalid_sparse,nullptr,1,ho,hout,work,sizeof(work));});
    expect(bad_layout.code==p::HvgCode::invalid_argument,"hvg sparse invalid layout");
    measure("hvg empty",[&]{return p::hvg_dense(empty,nullptr,1,ho,hout,nullptr,0);});
    return failures ? 1 : 0;
}
#else
int main() { return 77; }
#endif
