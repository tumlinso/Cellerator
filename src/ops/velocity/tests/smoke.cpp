#include <cstddef>
#include <Cellerator/ops/numeric/binary16.hh>
#include <cstdint>
import cellerator.velocity;
int main() {
    using namespace cellerator::velocity;
    using namespace velocitypp;
    uint16_t s[]={0x3c00,0x4000,0x4200},u[]={0x4000,0x4400,0x4600};
    uint16_t w[]={0x3c00,0,0,0,0x3c00,0,0,0,0x3c00};
    HalfMatrix a{anndatapp::Layout::dense,s,nullptr,nullptr,anndatapp::DType::int32,anndatapp::DType::int64,3,1,0,2,2};
    HalfMatrix b=a;b.values=u;
    HalfMatrix graph=a;graph.values=w;graph.columns=3;graph.row_stride_bytes=6;
    cellerator::velocity::Input in{{&a,nullptr},{&b,nullptr},{&graph,nullptr},{3,1,1}};
    if(!validate(in))return 1;
    float ms[3],mu[3],xs[3],xu[3];moments(in,0,1,ms,mu,xs,xu);
    for(int i=0;i<3;++i)if(ms[i]!=float(i+1) || mu[i]!=2*ms[i])return 2;
    float scalar[64]={};uint8_t mask[3]={};Workspace<float> work;
    RunOptions run;run.block_genes=1;
    if(!bind_workspace(in.domain,run,false,scalar,64,mask,3,work))return 3;
    float gamma[1],offset[1],r2[1];bool selected[1];uint8_t status[1];
    BasicFitResult<float> result{gamma,offset,r2,selected,status,1};
    FitOptions opts;opts.trimming=Trimming::none;opts.min_r2=-1;opts.min_ratio=-1;
    BasicFitStages<float> stages;if(!bind_stages(in.domain,work,opts,result,stages))return 4;
    BasicMomentBlock<float> block{0,1,3,1,ms,mu};auto fit=stages.fit(block);stages.emit(block,fit);stages.finish();
    if(gamma[0]!=2 || status[0]!=uint8_t(FitStatus::ok))return 5;
    uint16_t out[3];float scratch[3];residuals_f16(block,fit,scratch,out,1);
    for(auto v:out)if(v!=0)return 6;
    // Sparse half expression has duplicate entries; mixed FP64 expression is borrowed.
    uint16_t sv[]={0x3800,0x3800,0x4000,0x4200};
    int32_t si[]={0,0,1,2},so[]={0,4};
    HalfMatrix sparse; sparse.layout=anndatapp::Layout::csc;sparse.values=sv;
    sparse.indices=si;sparse.offsets=so;sparse.index_type=anndatapp::DType::int32;
    sparse.offset_type=anndatapp::DType::int32;sparse.rows=3;sparse.columns=1;sparse.nnz=4;
    double uv[]={2,5,6};auto owner=anndatapp::Array::borrow_dense(uv,{3,1},{1,1});
    auto view=owner.view();auto bytes=view.bytes();velocitypp::AcquiredMatrix acquired{&bytes,nullptr,anndatapp::DType::float64};
    uint16_t weights[]={0x3800,0x3800,0x3c00,0x3c00};int32_t wi[]={0,0,1,2},wo[]={0,2,3,4};
    HalfMatrix csr;csr.layout=anndatapp::Layout::csr;csr.values=weights;csr.indices=wi;csr.offsets=wo;
    csr.index_type=anndatapp::DType::int32;csr.offset_type=anndatapp::DType::int32;
    csr.rows=csr.columns=3;csr.nnz=4;
    cellerator::velocity::Input mixed{{&sparse,nullptr},{nullptr,&acquired},{&csr,nullptr},{3,1,1}};
    if(!validate(mixed))return 7;
    double ds[3],du[3],dss[3],dus[3],duu[3],sx[3],ux[3];
    second_moments(mixed,0,1,ds,du,dss,dus,duu,sx,ux);
    for(int i=0;i<3;++i)if(ds[i]!=i+1 || du[i]!=uv[i] || dss[i]!=(i+1)*(i+1) || dus[i]!=(i+1)*uv[i])return 8;
    // Drive both stochastic passes and compare the widened upstream reference.
    double sv64[]={1,2,3},wv64[]={1,0,0,0,1,0,0,0,1};
    auto sa=anndatapp::Array::borrow_dense(sv64,{3,1},{1,1});
    auto wa=anndatapp::Array::borrow_dense(wv64,{3,3},{3,1});
    auto sav=sa.view(),wav=wa.view();auto sb=sav.bytes(),wb=wav.bytes();
    velocitypp::Input upstream{{&sb,nullptr,anndatapp::DType::float64},acquired,{&wb,nullptr,anndatapp::DType::float64}};
    velocitypp::PreparationSize pn;if(!preparation_size(upstream,pn))return 11;
    size_t pi[64];velocitypp::Entry pe[64];velocitypp::PreparedInput prepared;
    if(!prepare(upstream,{pi,64,pe,64},prepared))return 12;
    double mem1[128]{},mem2[128]{};uint8_t masks1[3]{},masks2[3]{};
    velocitypp::Workspace<double> ws1,ws2;
    if(!bind_workspace(mixed.domain,run,true,mem1,128,masks1,3,ws1) ||
       !bind_workspace(prepared,run,true,mem2,128,masks2,3,ws2))return 13;
    double ig1[1],io1[1],ir1[1],g1[1],o1[1],o21[1],sc1[1],sc21[1],obj1[1];
    double ig2[1],io2[1],ir2[1],g2[1],o2[1],o22[1],sc2[1],sc22[1],obj2[1];
    bool sel1[1],sel2[1];uint8_t is1[1],is2[1],st1[1],st2[1];
    velocitypp::BasicFitResult<double> init1{ig1,io1,ir1,sel1,is1,1},init2{ig2,io2,ir2,sel2,is2,1};
    velocitypp::BasicStochasticFitResult<double> r1{init1,g1,o1,o21,sc1,sc21,obj1,st1,1},r2s{init2,g2,o2,o22,sc2,sc22,obj2,st2,1};
    velocitypp::StochasticOptions stochastic;stochastic.initialization=opts;
    velocitypp::BasicStochasticFitStages<double> stages2;
    if(!bind_stages(mixed.domain,ws1,stochastic,r1,stages2))return 14;
    moments(mixed,0,1,ws1.ms,ws1.mu,sx,ux);
    velocitypp::BasicMomentBlock<double> initial{0,1,3,1,ws1.ms,ws1.mu};
    auto initfit=stages2.initial_fit(initial);stages2.accept_initialization(initial,initfit);stages2.resolve_selection();
    second_moments(mixed,0,1,ws1.ms,ws1.mu,ws1.mss,ws1.mus,(double*)nullptr,sx,ux);
    velocitypp::BasicStochasticMomentBlock<double> sm{0,1,3,1,ws1.ms,ws1.mu,ws1.mss,ws1.mus,nullptr};
    auto scales=stages2.scales(sm);auto sf=stages2.fit(sm,scales);stages2.emit(sm,sf);stages2.finish();
    double output1[3],output2[3];velocitypp::reference::stochastic_residuals(sm,sf,output1,(double*)nullptr,1);
    if(!velocitypp::reference::stochastic_into(prepared,ws2,r2s,output2,3,(double*)nullptr,0,1,stochastic))return 15;
    if(st1[0]!=st2[0] || is1[0]!=is2[0] || sel1[0]!=sel2[0] || st1[0]!=uint8_t(StochasticStatus::ok))return 16;
    for(int i=0;i<3;++i)if(output1[i]!=output2[i])return 17;
    velocitypp::BasicMomentBlock<double> mb{0,1,3,1,ds,du};
    PromotionMoments<double> promotion{&mixed,&mb,nullptr,sx,ux};
    velocitypp::precision::OriginalColumn original{uv,uv,ptrdiff_t(sizeof(double)),ptrdiff_t(sizeof(double)),anndatapp::DType::float64,anndatapp::DType::float64,3};
    if(replace64(&promotion,0,0,original,false)!=velocitypp::precision::PrecisionCode::ok)return 9;
    for(int i=0;i<3;++i)if(ds[i]!=uv[i] || du[i]!=uv[i])return 10;
    return 0;
}
