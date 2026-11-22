#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>

#include <Cellerator/memory/view.hh>
#include <Cellerator/tensor/base/compressed.hh>

import preprocess;

static std::size_t allocations;
void* operator new(std::size_t n) { ++allocations; if (void* p = std::malloc(n)) return p; throw std::bad_alloc(); }
void* operator new[](std::size_t n) { ++allocations; if (void* p = std::malloc(n)) return p; throw std::bad_alloc(); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

bool near(double a, double b, double e = 1e-10) { return std::abs(a-b) <= e; }

int main() {
    using cellerator::memory::const_matrix_view;
    using cellerator::memory::matrix_view;
    using cellerator::tensor::base::CompressedAxis;
    using cellerator::tensor::base::CompressedView;
    const double values[]={1,3,0,0,2,6};
    const_matrix_view<double> in{values,3,2,2};
    double factors[3], work[3]; uint8_t included[2];
    preprocess::NormalizationBuffers norm{factors,3,included,2,work,3,0};
    preprocess::NormalizationOptions automatic{};
    auto before = allocations;
    auto status = preprocess::normalization_factors(in,automatic,norm);
    if (!status || allocations != before || !near(norm.target_sum,6) || !near(factors[0],2.0/3) || factors[1] != 0 || !near(factors[2],4.0/3) || !included[0] || !included[1]) return 1;
    double automatic_factors[]={factors[0],factors[1],factors[2]};
    double automatic_target=norm.target_sum;

    float out[6]; matrix_view<float> dst{out,3,2,2};
    preprocess::NormalizationOptions fixed{true,12,false,.05};
    before = allocations;
    status = preprocess::normalize_into(in,dst,fixed,norm);
    if (!status || allocations != before || !near(norm.target_sum,12) || !near(factors[0],1.0/3) || !near(factors[1],0) || !near(factors[2],2.0/3)) return 2;
    for (size_t i = 0; i < 6; ++i) if (!near(out[i],factors[i/2] > 0 ? values[i]/factors[i/2] : values[i])) return 3;

    double tf[3], tw[3]; uint8_t ti[2];
    preprocess::NormalizationBuffers tnorm{tf,3,ti,2,tw,3,0};
    auto trusted_auto=automatic; trusted_auto.trusted=true;
    before=allocations; auto trusted_status=preprocess::normalization_factors(in,trusted_auto,tnorm);
    if(!trusted_status || allocations!=before || !near(tnorm.target_sum,automatic_target)) return 23;
    for(size_t i=0;i<3;++i) if(!near(tf[i],automatic_factors[i])) return 24;
    auto trusted_fixed=fixed; trusted_fixed.trusted=true;
    float out_trusted[6]; matrix_view<float> dtrusted{out_trusted,3,2,2};
    before=allocations; trusted_status=preprocess::normalize_into(in,dtrusted,trusted_fixed,tnorm);
    if(!trusted_status || allocations!=before || !near(tnorm.target_sum,norm.target_sum)) return 25;
    for(size_t i=0;i<6;++i) if(!near(out_trusted[i],out[i]) || !near(tf[i/2],factors[i/2])) return 26;

    double custom_factors[]={0,2,4}; double custom_out[6]; matrix_view<double> custom_dst{custom_out,3,2,2};
    before=allocations; status=preprocess::apply_normalization(in,custom_dst,custom_factors,3);
    if(!status || allocations!=before) return 27;
    for(size_t i=0;i<6;++i) if(!near(custom_out[i],custom_factors[i/2] ? values[i]/custom_factors[i/2] : values[i])) return 28;
    status=preprocess::apply_normalization(in,custom_dst,custom_factors,2);
    if(status.code!=preprocess::QcCode::insufficient_capacity || status.required!=3) return 29;
    double bad_factors[]={1,-1,1};
    status=preprocess::apply_normalization(in,custom_dst,bad_factors,3);
    if(status.code!=preprocess::QcCode::invalid_value || status.index!=1) return 30;
    preprocess::NormalizationOptions exclusion{true,1,true,.8};
    const double high[]={99,1};
    const_matrix_view<double> hv{high,1,2,2}; double hf[1], hw[1]; uint8_t hi[2];
    preprocess::NormalizationBuffers hb{hf,1,hi,2,hw,1,0};
    before=allocations; status=preprocess::normalization_factors(hv,exclusion,hb);
    if (!status || allocations != before || hi[0] || !hi[1] || !near(hf[0],1) || !near(hb.target_sum,1)) return 4;

    double logarithm[6]; matrix_view<double> ld{logarithm,3,2,2};
    before=allocations; status=preprocess::log1p_into(in,ld);
    if (!status || allocations != before) return 5;
    for(size_t i=0;i<6;++i) if(!near(logarithm[i],std::log1p(values[i]))) return 6;
    double trusted_log[6]; matrix_view<double> tld{trusted_log,3,2,2};
    before=allocations; trusted_status=preprocess::log1p_into(in,tld,0,true);
    if(!trusted_status || allocations!=before) return 42;
    for(size_t i=0;i<6;++i) if(!near(trusted_log[i],logarithm[i])) return 43;
    status=preprocess::log1p_into(in,ld,1);
    if(status.code != preprocess::QcCode::invalid_argument) return 7;

    double scaled[6]; matrix_view<double> sdout{scaled,3,2,2};
    uint64_t stored[2]; double mean[2], deviation[2]; uint8_t selected[]={1,0,1};
    preprocess::ScaleOptions so{true,true,1.0,selected};
    before=allocations; status=preprocess::scale_into(in,sdout,so,mean,deviation,2,stored,2);
    if(!status || allocations!=before || !near(mean[0],1.5) || !near(mean[1],4.5) || !near(deviation[0],std::sqrt(.5)) || !near(deviation[1],std::sqrt(4.5))) return 8;
    if(!near(scaled[0],-std::sqrt(.5)) || !near(scaled[1],-std::sqrt(.5)) || scaled[2]!=0 || scaled[3]!=0 || !near(scaled[4],std::sqrt(.5)) || !near(scaled[5],std::sqrt(.5))) return 9;

    auto trusted_scale=so; trusted_scale.trusted=true;
    double scale_trusted[6]; matrix_view<double> st{scale_trusted,3,2,2};
    double mt[2], dt[2]; uint64_t wt[2];
    before=allocations; trusted_status=preprocess::scale_into(in,st,trusted_scale,mt,dt,2,wt,2);
    if(!trusted_status || allocations!=before) return 31;
    for(size_t i=0;i<6;++i) if(!near(scale_trusted[i],scaled[i]) || !near(mt[i%2],mean[i%2]) || !near(dt[i%2],deviation[i%2])) return 32;

    const_matrix_view<double,ptrdiff_t> ri{values+4,3,2,-2};
    double revout[6]; matrix_view<double> rd{revout,3,2,2};
    before=allocations; status=preprocess::log1p_into(ri,rd);
    if(!status || allocations!=before) return 10;
    if(!near(revout[0],std::log1p(2)) || !near(revout[1],std::log1p(6)) || !near(revout[2],0) || !near(revout[3],0) || !near(revout[4],std::log1p(1)) || !near(revout[5],std::log1p(3))) return 11;
    const_matrix_view<double,ptrdiff_t> reversed_columns{values+1,3,2,2,{},-1};
    matrix_view<double,ptrdiff_t> reversed_output{revout+4,3,2,-2};
    before=allocations; status=preprocess::log1p_into(reversed_columns,reversed_output);
    if(!status || allocations!=before) return 54;
    for(size_t i=0;i<3;++i) for(size_t j=0;j<2;++j)
        if(!near(revout[2*(2-i)+j],std::log1p(values[2*i+1-j]))) return 55;

    const double cv[]={1,3,2,6};
    const int32_t ci[]={0,1,0,1};
    const int64_t co[]={0,2,2,4};
    CompressedView<int32_t,const double,int64_t> cs{{3,2,4,CompressedAxis::row,co,ci},cv};
    float csv[4];
    const int32_t normalized_indices[]={0,1,0,1};
    const int64_t normalized_offsets[]={0,2,2,4};
    CompressedView<int32_t,float,int64_t> csout{{3,2,4,CompressedAxis::row,normalized_offsets,normalized_indices},csv};
    before=allocations; status=preprocess::normalize_into(cs,csout,fixed,norm);
    if(!status || allocations!=before || !near(factors[0],1.0/3) || !near(factors[1],0) || !near(factors[2],2.0/3)) return 12;
    if(!near(csv[0],3) || !near(csv[1],9) || !near(csv[2],3) || !near(csv[3],9)) return 13;

    auto trusted_csr=automatic; trusted_csr.target_is_set=true; trusted_csr.target_sum=12; trusted_csr.trusted=true;
    float tcval[4];
    CompressedView<int32_t,float,int64_t> tco{csout.structure,tcval}; double cf[3],cw[3]; uint8_t csel[2];
    preprocess::NormalizationBuffers cb{cf,3,csel,2,cw,3,0};
    before=allocations; trusted_status=preprocess::normalize_into(cs,tco,trusted_csr,cb);
    if(!trusted_status || allocations!=before || !near(cb.target_sum,12)) return 33;
    for(int i=0;i<4;++i) if(!near(tcval[i],csv[i])) return 34;
    double sparse_factors[]={0,2,4}, sav[4];
    CompressedView<int32_t,double,int64_t> sparse_apply{cs.structure,sav};
    before=allocations;
    status=preprocess::apply_normalization(cs,sparse_apply,sparse_factors,3);
    if(!status || allocations!=before) return 35;
    if(!near(sav[0],1) || !near(sav[1],3) || !near(sav[2],.5) || !near(sav[3],1.5)) return 36;
    double dense_sparse[6]; matrix_view<double> dsout{dense_sparse,3,2,2};
    before=allocations; status=preprocess::log1p_into(cs,dsout);
    if(!status || allocations!=before) return 14;
    if(!near(dense_sparse[0],std::log(2)) || !near(dense_sparse[1],std::log(4)) || !near(dense_sparse[2],0) || !near(dense_sparse[3],0) || !near(dense_sparse[4],std::log(3)) || !near(dense_sparse[5],std::log(7))) return 15;
    double trusted_sparse_log[6]; matrix_view<double> tsld{trusted_sparse_log,3,2,2};
    before=allocations; trusted_status=preprocess::log1p_into(cs,tsld,0,true);
    if(!trusted_status || allocations!=before) return 44;
    for(int i=0;i<6;++i) if(!near(trusted_sparse_log[i],dense_sparse[i])) return 45;

    const double sv[]={1,2,3,6};
    const int64_t si[]={0,2,0,2};
    const int32_t sofs[]={0,2,4};
    CompressedView<int64_t,const double,int32_t> sci{{3,2,4,CompressedAxis::column,sofs,si},sv};
    double csc_factors[3], csc_work[3]; uint8_t csc_included[2];
    preprocess::NormalizationBuffers csc_norm{csc_factors,3,csc_included,2,csc_work,3,0};
    auto checked_csc=automatic; before=allocations; status=preprocess::normalization_factors(sci,checked_csc,csc_norm);
    if(!status || allocations!=before || !near(csc_factors[0],2.0/3) || csc_factors[1]!=0 || !near(csc_factors[2],4.0/3)) return 37;
    checked_csc.trusted=true; double csc_tf[3], csc_tw[3]; uint8_t csc_ti[2];
    preprocess::NormalizationBuffers csc_tnorm{csc_tf,3,csc_ti,2,csc_tw,3,0};
    before=allocations; status=preprocess::normalization_factors(sci,checked_csc,csc_tnorm);
    if(!status || allocations!=before || !near(csc_tnorm.target_sum,csc_norm.target_sum)) return 38;
    for(int i=0;i<3;++i) if(!near(csc_tf[i],csc_factors[i])) return 39;
    before=allocations; status=preprocess::log1p_into(sci,dsout);
    if(!status || allocations!=before) return 16;
    if(!near(dense_sparse[0],std::log(2)) || !near(dense_sparse[1],std::log(4)) || !near(dense_sparse[2],0) || !near(dense_sparse[3],0) || !near(dense_sparse[4],std::log(3)) || !near(dense_sparse[5],std::log(7))) return 17;
    double trusted_csc_log[6]; matrix_view<double> tccd{trusted_csc_log,3,2,2};
    before=allocations; trusted_status=preprocess::log1p_into(sci,tccd,0,true);
    if(!trusted_status || allocations!=before) return 46;
    for(int i=0;i<6;++i) if(!near(trusted_csc_log[i],dense_sparse[i])) return 47;
    double csc_values[4];
    CompressedView<int64_t,double,int32_t> cso{sci.structure,csc_values};
    before=allocations; status=preprocess::log1p_into(sci,cso);
    if(!status || allocations!=before) return 18;
    if(!near(csc_values[0],std::log(2)) || !near(csc_values[1],std::log(3)) || !near(csc_values[2],std::log(4)) || !near(csc_values[3],std::log(7))) return 19;

    before=allocations; status=preprocess::normalize_into(in,dst,{true,0,false,.05},norm);
    if(status.code!=preprocess::QcCode::invalid_argument || allocations!=before) return 20;
    preprocess::NormalizationBuffers short_norm{factors,2,included,2,work,3,0};
    status=preprocess::normalization_factors(in,automatic,short_norm);
    if(status.code!=preprocess::QcCode::insufficient_capacity || status.required!=3) return 21;
    status=preprocess::scale_into(cs,csout,{true,false,0,nullptr},mean,deviation,2,stored,2);
    if(status.code!=preprocess::QcCode::invalid_argument) return 22;

    double overlap_data[]={1,2,3,4,5,6};
    const_matrix_view<double> overlap_src{overlap_data,2,2,2};
    double sentinels_zero[]={91,92,93,94}, sentinels_one[]={91,92,93,94};
    matrix_view<double> oz{sentinels_zero,2,2,0}, oo{sentinels_one,2,2,1};
    before=allocations; status=preprocess::log1p_into(overlap_src,oz);
    if(status.code!=preprocess::QcCode::invalid_argument || allocations!=before || sentinels_zero[0]!=91 || sentinels_zero[1]!=92 || sentinels_zero[2]!=93 || sentinels_zero[3]!=94) return 40;
    status=preprocess::log1p_into(overlap_src,oo);
    if(status.code!=preprocess::QcCode::invalid_argument || allocations!=before || sentinels_one[0]!=91 || sentinels_one[1]!=92 || sentinels_one[2]!=93 || sentinels_one[3]!=94) return 41;
    double partial[]={1,2,3,4,5};
    const_matrix_view<double> partial_src{partial,2,2,2};
    matrix_view<double> partial_dst{partial+1,2,2,2};
    before=allocations;
    status=preprocess::log1p_into(partial_src,partial_dst);
    if(status.code!=preprocess::QcCode::invalid_argument || allocations!=before || partial[0]!=1 || partial[1]!=2 || partial[2]!=3 || partial[3]!=4 || partial[4]!=5) return 50;
    matrix_view<double> exact_dst{overlap_data,2,2,2};
    before=allocations;
    status=preprocess::log1p_into(overlap_src,exact_dst);
    if(!status || allocations!=before || !near(overlap_data[0],std::log1p(1)) || !near(overlap_data[3],std::log1p(4))) return 51;

    const double square_in[]={1,2};
    const int32_t square_indices[]={0,1}, square_offsets[]={0,1,2};
    CompressedView<int32_t,const double> square_src{{2,2,2,CompressedAxis::row,square_offsets,square_indices},square_in};
    double square_out[]={81,82};
    CompressedView<int32_t,double> square_dst{{2,2,2,CompressedAxis::column,square_offsets,square_indices},square_out};
    status=preprocess::log1p_into(square_src,square_dst);
    if(status.code!=preprocess::QcCode::invalid_argument || square_out[0]!=81 || square_out[1]!=82) return 52;
    double changed_out[]={71,72};
    const int32_t changed_indices[]={1,0};
    CompressedView<int32_t,double> changed_dst{{2,2,2,CompressedAxis::row,square_offsets,changed_indices},changed_out};
    status=preprocess::log1p_into(square_src,changed_dst);
    if(status.code!=preprocess::QcCode::invalid_structure || changed_out[0]!=71 || changed_out[1]!=72) return 53;

    const int32_t invalid_offsets[]={0,1,3}, oversized_indices[]={0,0,1};
    const double oversized_values[]={1,2,3};
    CompressedView<int32_t,const double> invalid_source{
        {2,2,2,CompressedAxis::row,invalid_offsets,oversized_indices},oversized_values};
    double guarded_out[]={61,62,63,64};
    matrix_view<double> guarded{guarded_out,2,2,2};
    status=preprocess::log1p_into(invalid_source,guarded);
    if(status.code!=preprocess::QcCode::invalid_structure || guarded_out[0]!=61 ||
       guarded_out[1]!=62 || guarded_out[2]!=63 || guarded_out[3]!=64) return 56;
    const double dense_input[]={1,2,3,4};
    const_matrix_view<double,ptrdiff_t> invalid_stride{dense_input,2,2,PTRDIFF_MAX};
    status=preprocess::log1p_into(invalid_stride,guarded);
    if(status.code!=preprocess::QcCode::invalid_argument || guarded_out[0]!=61 ||
       guarded_out[1]!=62 || guarded_out[2]!=63 || guarded_out[3]!=64) return 57;
    auto device_only=invalid_stride;
    device_only.row_stride=2;
    device_only.where.kind=cellerator::memory::domain::device;
    status=preprocess::log1p_into(device_only,guarded);
    if(status.code!=preprocess::QcCode::invalid_argument || guarded_out[0]!=61 ||
       guarded_out[1]!=62 || guarded_out[2]!=63 || guarded_out[3]!=64) return 58;
    return 0;
}
