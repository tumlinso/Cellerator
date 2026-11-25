#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>

#include <Cellerator/memory/view.hh>
#include <Cellerator/tensor/base/compressed.hh>

import cellerator.preprocess;

namespace { bool counting = false; size_t allocations = 0; }
void* operator new(std::size_t n) { if (counting) ++allocations; if (void* p = std::malloc(n ? n : 1)) return p; throw std::bad_alloc(); }
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

namespace {
bool close(double a,double b) { return std::fabs(a-b)<1e-12; }
int fail(const char* msg) { std::fprintf(stderr,"%s\n",msg); return 1; }

template<class X> int exercise(const X& x) {
    using namespace cellerator::preprocess;
    double cells[3], genes[3], mean[3], dropout[3], groups[3], pct[3], top[6], logs[3], scratch[3];
    uint64_t cell_detected[3],gene_detected[3],stored[3]; size_t observations=0;
    const uint8_t mask[3]={1,0,1}; const size_t tops[2]={1,2};
    QcOptions opt{mask,1,tops,2};
    QcBuffers out{cells,cell_detected,3,genes,gene_detected,3,mean,dropout,nullptr,nullptr,nullptr,logs,groups,pct,nullptr,3,top,6};
    allocations=0; counting=true;
    auto status=qc(x,opt,out,scratch,3);
    counting=false;
    if (!status || allocations) return fail("qc status or allocation");
    if (!close(cells[0],4)||!close(cells[2],6)||!close(genes[0],3)||!close(genes[1],4)||!close(genes[2],3)) return fail("qc totals");
    if (cell_detected[0]!=2 || cell_detected[1]!=0 || gene_detected[0]!=2 || !close(groups[0],4) || !close(groups[2],2)) return fail("qc detection/groups");
    if (!close(top[0],75)||!close(top[1],100)||!close(top[4],200.0/3)||!close(top[5],100)) return fail("qc top");
    if (!std::isnan(top[2]) || !std::isnan(pct[1])) return fail("zero row");
    double var[3];
    allocations=0; counting=true;
    status=feature_stats(x,nullptr,mean,var,3,stored,3,&observations);
    counting=false;
    if (!status || allocations || observations!=3) return fail("stats status or allocation");
    if (!close(mean[0],1)||!close(var[0],1)||!close(mean[1],4.0/3)||!close(var[1],16.0/3)||!close(var[2],3)) return fail("stats values");
    uint8_t keep[3]; FilterBounds b; b.has_min_counts=true; b.min_counts=5;
    allocations=0; counting=true; status=filter(cells,cell_detected,3,b,keep,3); counting=false;
    if (!status || allocations || keep[0] || keep[1] || !keep[2]) return fail("filter");
    status=qc_check(3,3,opt,out,2); if (status.code!=QcCode::insufficient_capacity || status.required!=3) return fail("scratch capacity");
    auto attempt = [&](const QcBuffers& buffers,double* work,size_t work_size) {
        return qc_accumulate(x,0,3,opt,buffers,work,work_size);
    };
    auto unchanged = [&] {
        for (size_t i=0;i<3;++i)
            if (cells[i]!=31 || genes[i]!=32 || groups[i]!=33 || scratch[i]!=34 || cell_detected[i]!=35 || gene_detected[i]!=36) return false;
        for (double v : top) if (v!=37) return false;
        return true;
    };
    for (size_t i=0;i<3;++i) { cells[i]=31; genes[i]=32; groups[i]=33; scratch[i]=34; cell_detected[i]=35; gene_detected[i]=36; }
    for (double& v : top) v=37;
    QcBuffers short_out=out; short_out.cells_capacity=2;
    status=attempt(short_out,scratch,3); if (status.code!=QcCode::insufficient_capacity || status.required!=3 || !unchanged()) return fail("partial cell capacity changed outputs");
    short_out=out; short_out.genes_capacity=2;
    status=attempt(short_out,scratch,3); if (status.code!=QcCode::insufficient_capacity || status.required!=3 || !unchanged()) return fail("partial gene capacity changed outputs");
    short_out=out; short_out.group_capacity=2;
    status=attempt(short_out,scratch,3); if (status.code!=QcCode::insufficient_capacity || status.required!=3 || !unchanged()) return fail("partial group capacity changed outputs");
    short_out=out; short_out.top_capacity=5;
    status=attempt(short_out,scratch,3); if (status.code!=QcCode::insufficient_capacity || status.required!=6 || !unchanged()) return fail("partial top capacity changed outputs");
    status=attempt(out,scratch,2); if (status.code!=QcCode::insufficient_capacity || status.required!=3 || !unchanged()) return fail("partial scratch capacity changed outputs");
    status=attempt(out,nullptr,3); if (status.code!=QcCode::invalid_argument || !unchanged()) return fail("partial null scratch changed outputs");
    allocations=0; counting=true;
    status=qc_validate(x,true);
    if (status) {
        qc_zero(3,3,1,out);
        status=qc_accumulate_unchecked(x,0,3,opt,out,scratch);
        if (status) qc_finish(3,3,opt,out);
    }
    if (status) status=feature_stats_unchecked(x,nullptr,mean,var,3,stored,3,&observations);
    counting=false;
    if (!status || allocations || !close(var[2],3) || !close(top[0],75)) return fail("unchecked stages or allocation");
    double partial_total[]={1,2,3}, part2[]={4,5,6}; uint64_t partial_detected[]={1,2,3}, part2_detected[]={4,5,6};
    allocations=0; counting=true; qc_merge(3,partial_total,partial_detected,part2,part2_detected); counting=false;
    if (allocations || partial_total[2]!=9 || partial_detected[2]!=9) return fail("partial merge");
    double ma[3],m2a[3],mb[3],m2b[3],merged_variance[3]; uint64_t counta[3],countb[3];
    StatsPartial pa{0,0,ma,m2a,3},pb{0,0,mb,m2b,3};
    allocations=0; counting=true;
    status=stats_partial(x,0,1,nullptr,pa,counta,3);
    if (status) status=stats_partial(x,1,3,nullptr,pb,countb,3);
    if (status) status=stats_merge(pa,pb);
    if (status) status=stats_finish(pa,merged_variance,3);
    counting=false;
    if (!status || allocations || pa.observations!=3 || !close(pa.mean[0],1) || !close(merged_variance[0],1) || !close(merged_variance[1],16.0/3) || !close(merged_variance[2],3)) return fail("partial stats and merge");
    const uint8_t selected[]={1,0,1};
    allocations=0; counting=true;
    status=feature_stats(x,selected,mean,var,3,stored,3,&observations);
    counting=false;
    if (!status || allocations || observations!=2 || !close(mean[0],1.5) || !close(var[0],.5) || !close(mean[1],2) || !close(var[1],8)) return fail("masked stats");
    const uint8_t none[]={0,0,0};
    status=stats_partial(x,0,1,none,pa,counta,3);
    if (status) status=stats_partial(x,1,3,none,pb,countb,3);
    if (status) status=stats_merge(pa,pb);
    if (status) status=stats_finish(pa,merged_variance,3);
    if (!status || pa.observations || !std::isnan(pa.mean[0]) || !std::isnan(merged_variance[0])) return fail("empty partial stats finalization");
    status=feature_stats(x,none,mean,var,3,stored,3,&observations);
    if (!status || observations || !std::isnan(mean[0]) || !std::isnan(var[0])) return fail("empty stats selection");
    const uint8_t invalid_mask[]={1,3,0};
    status=feature_stats(x,invalid_mask,mean,var,3,stored,3,&observations);
    if (status.code!=QcCode::invalid_value || status.index!=1) return fail("invalid mask");
    return 0;
}
}
int main() {
    using cellerator::memory::const_matrix_view;
    using cellerator::tensor::base::CompressedAxis;
    using cellerator::tensor::base::CompressedView;
    const double values[]={1,0,3,0,0,0,2,4,0};
    if (int r=exercise(const_matrix_view<double>{values,3,3,3})) return r;

    const double vv[]={1,0,3,2,4};
    uint32_t ix[]={0,1,2,0,1};
    const uint64_t off[]={0,3,3,5};
    CompressedView<uint32_t,const double,uint64_t> csr{{3,3,5,CompressedAxis::row,off,ix},vv};
    if (int r=exercise(csr)) return r;
    ix[1]=0;
    auto invalid=cellerator::preprocess::qc_validate(csr,true);
    if (invalid.code!=cellerator::preprocess::QcCode::invalid_structure || invalid.index!=1) return fail("duplicate sparse index");

    const float cv[]={1,2,4,3,0};
    const uint64_t ci[]={0,2,2,0,2};
    const uint32_t co[]={0,2,3,5};
    CompressedView<uint64_t,const float,uint32_t> csc{{3,3,5,CompressedAxis::column,co,ci},cv};
    if (int r=exercise(csc)) return r;

    const float nonintegral_index[] = {0};
    const int64_t row_offsets[] = {0,1};
    CompressedView<float,const float,int64_t> invalid_index{
        {1,1,1,CompressedAxis::row,row_offsets,nonintegral_index},cv};
    if (cellerator::preprocess::qc_validate(invalid_index,true)) return fail("noninteger sparse index");
    const bool bool_offsets[] = {false,true};
    CompressedView<uint64_t,const float,bool> invalid_offsets{
        {1,1,1,CompressedAxis::row,bool_offsets,ci},cv};
    if (cellerator::preprocess::qc_validate(invalid_offsets,true)) return fail("boolean sparse offsets");

    if constexpr (sizeof(size_t)>4) {
        const uint64_t columns = uint64_t(1)<<32;
        const int32_t negative[] = {-1};
        const uint32_t high[] = {UINT32_MAX};
        CompressedView<int32_t,const float,int64_t> signed_index{
            {1,columns,1,CompressedAxis::row,row_offsets,negative},cv};
        CompressedView<uint32_t,const float,int64_t> unsigned_index{
            {1,columns,1,CompressedAxis::row,row_offsets,high},cv};
        auto s = cellerator::preprocess::qc_validate(signed_index,true);
        if (s.code != cellerator::preprocess::QcCode::invalid_structure || !cellerator::preprocess::qc_validate(unsigned_index,true))
            return fail("sparse index signedness");
    }

    const double reverse_data[]={2,4,0,0,0,0,1,0,3};
    return exercise(const_matrix_view<double,ptrdiff_t>{reverse_data+6,3,3,-3});
}
