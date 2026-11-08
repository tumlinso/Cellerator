#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

import velocity.cuda;

namespace {
namespace ac=anndata::cuda;
namespace vc=velocity::cuda;
using ac::ScalarType;
int failures=0;
void expect(bool yes,const char* message) {if(!yes){std::cerr<<"FAIL: "<<message<<'\n';++failures;}}
void runtime(cudaError_t e) {if(e)throw std::runtime_error(cudaGetErrorString(e));}
template<class F> void invalid(F f,const char* message) {
    try {f();expect(false,message);} catch(const std::invalid_argument&) {} catch(const std::length_error&) {}
}
struct DeviceMatrix {
    ac::DeviceBuffer values,indices,offsets;
    vc::MatrixView view;
};
DeviceMatrix upload(anndata::ArrayView source,ac::Stream stream) {
    DeviceMatrix out;auto n=source.extent(0),g=source.extent(1);
    if(source.layout()==anndata::Layout::dense) {
        auto b=source.bytes();auto rs=b.stride_bytes(0),cs=b.stride_bytes(1);
        auto r=n?std::ptrdiff_t(n-1)*rs:0,c=g?std::ptrdiff_t(g-1)*cs:0;
        auto lo=std::min(std::ptrdiff_t(0),r)+std::min(std::ptrdiff_t(0),c);
        auto hi=std::max(std::ptrdiff_t(0),r)+std::max(std::ptrdiff_t(0),c);
        auto bytes=n && g?std::size_t(hi-lo)+b.element_size():0;
        out.values=ac::DeviceBuffer(bytes,stream.device);
        if(bytes)ac::copy_to_device(out.values.span(),0,b.data()+lo,bytes,stream);
        out.view=ac::DenseView{out.values.span(),std::uint64_t(-lo),n,g,rs/std::ptrdiff_t(b.element_size()),cs/std::ptrdiff_t(b.element_size()),ac::scalar_type(source.dtype())};
    } else {
        auto b=source.sparse_bytes();auto major=source.layout()==anndata::Layout::csr?n:g;
        auto values=b.nnz()*anndata::dtype_size(source.dtype()),indices=b.nnz()*b.index_size(),offsets=(major+1)*b.offset_size();
        out.values=ac::DeviceBuffer(values,stream.device);out.indices=ac::DeviceBuffer(indices,stream.device);out.offsets=ac::DeviceBuffer(offsets,stream.device);
        if(values)ac::copy_to_device(out.values.span(),0,b.values(),values,stream);
        if(indices)ac::copy_to_device(out.indices.span(),0,b.indices(),indices,stream);
        ac::copy_to_device(out.offsets.span(),0,b.offsets(),offsets,stream);
        out.view=ac::SparseView{out.values.span(),out.indices.span(),out.offsets.span(),n,g,b.nnz(),
            ac::scalar_type(source.dtype()),ac::scalar_type(source.index_dtype()),ac::scalar_type(source.offset_dtype()),
            source.layout()==anndata::Layout::csr?ac::SparseLayout::csr:ac::SparseLayout::csc};
    }
    return out;
}
struct Output {
    ac::DeviceBuffer owner;
    ac::DenseView view;
    Output(std::size_t n,std::size_t g,ScalarType type,ac::Stream stream,bool reverse=false)
        :owner(n*g*ac::scalar_size(type),stream.device),
         view{owner.span(),reverse && n?(n-1)*g*ac::scalar_size(type):0,n,g,reverse?-std::int64_t(g):std::int64_t(g),1,type} {}
};
std::vector<double> read(ac::DenseView view,ac::Stream stream) {
    Output converted(view.rows,view.columns,ScalarType::float64,stream);
    ac::cast(converted.view,view,stream);std::vector<double> out(view.rows*view.columns);
    if(!out.empty())ac::copy_to_host(out.data(),converted.owner.span(),0,out.size()*sizeof(double),stream);
    ac::synchronize(stream);return out;
}
std::vector<double> host(anndata::ArrayView view) {
    std::vector<double> out;auto b=view.bytes();out.reserve(view.extent(0)*(view.rank()==2?view.extent(1):1));
    auto n=view.extent(0),g=view.rank()==2?view.extent(1):1;
    for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<g;++j) {
        auto* p=b.data()+i*b.stride_bytes(0)+(view.rank()==2?std::ptrdiff_t(j)*b.stride_bytes(1):0);
        if(view.dtype()==anndata::DType::float64){double x;std::memcpy(&x,p,sizeof x);out.push_back(x);}
        else if(view.dtype()==anndata::DType::float32){float x;std::memcpy(&x,p,sizeof x);out.push_back(x);}
        else if(view.dtype()==anndata::DType::float16){anndata::Float16 x;std::memcpy(&x,p,sizeof x);out.push_back(x.to_float());}
        else {std::uint8_t x;std::memcpy(&x,p,1);out.push_back(x);}
    }
    return out;
}
void close(const std::vector<double>& actual,const std::vector<double>& expected,double tolerance,const char* message) {
    expect(actual.size()==expected.size(),message);if(actual.size()!=expected.size())return;
    double scale=1;for(double x:expected)scale=std::max(scale,std::abs(x));
    for(std::size_t i=0;i<actual.size();++i)if(std::abs(actual[i]-expected[i])>tolerance*(scale+std::abs(expected[i]))) {
        std::cerr<<message<<" index="<<i<<" actual="<<actual[i]<<" expected="<<expected[i]<<'\n';expect(false,message);return;
    }
}
template<class T> anndata::Array dense(std::size_t n,std::size_t g,const std::vector<double>& values) {
    auto out=anndata::Array::empty<T>({n,g});auto* p=out.edit().template dense<T>().data();
    for(std::size_t i=0;i<values.size();++i) {
        if constexpr(std::is_same_v<T,anndata::Float16>)p[i]=anndata::Float16::from_double(values[i]);else p[i]=T(values[i]);
    }
    return out;
}
template<class T> anndata::Array sparse(std::size_t n,std::size_t g,const std::vector<double>& values,anndata::Layout layout) {
    auto major=layout==anndata::Layout::csr?n:g,minor=layout==anndata::Layout::csr?g:n;
    std::vector<T> data;std::vector<std::uint8_t> indices;std::vector<std::uint64_t> offsets{0};
    for(std::size_t a=0;a<major;++a) {
        // Reverse physical order, split every nonzero into duplicate coordinates.
        for(std::size_t j=minor;j>0;--j) {
            auto b=j-1,r=layout==anndata::Layout::csr?a:b,c=layout==anndata::Layout::csr?b:a;
            auto x=values[r*g+c];if(!x)continue;
            if constexpr(std::is_same_v<T,anndata::Float16>) {data.push_back(anndata::Float16::from_double(x*.5));data.push_back(anndata::Float16::from_double(x*.5));}
            else {data.push_back(T(x*.5));data.push_back(T(x*.5));}
            indices.push_back(std::uint8_t(b));indices.push_back(std::uint8_t(b));
        }
        offsets.push_back(data.size());
    }
    auto d=anndata::Array::empty<T>({data.size()});auto i=anndata::Array::empty<std::uint8_t>({indices.size()});auto o=anndata::Array::empty<std::uint64_t>({offsets.size()});
    std::copy(data.begin(),data.end(),d.edit().template dense<T>().data());
    std::copy(indices.begin(),indices.end(),i.edit().dense<std::uint8_t>().data());
    std::copy(offsets.begin(),offsets.end(),o.edit().dense<std::uint64_t>().data());
    return anndata::Array::from_sparse_buffers(layout,std::move(d),std::move(i),std::move(o),n,g);
}
template<class T> void compare_case(ac::Stream stream,bool compressed) {
    constexpr std::size_t n=18,g=5;std::vector<double> s(n*g),u(n*g),w(n*n);
    for(std::size_t i=0;i<n;++i) {
        w[i*n+i]=.6;w[i*n+(i+1)%n]=.4;
        for(std::size_t j=0;j<g;++j) {double x=1+double((i*7+j*3)%19)/4;s[i*g+j]=x;u[i*g+j]=(.4+.2*j)*x+.13*double((i+j)%5)+.2*j;}
    }
    auto hs=compressed?sparse<T>(n,g,s,anndata::Layout::csr):dense<T>(n,g,s);
    auto hu=compressed?sparse<T>(n,g,u,anndata::Layout::csc):dense<T>(n,g,u);
    auto hw=compressed?sparse<T>(n,n,w,anndata::Layout::csc):dense<T>(n,n,w);
    std::vector<std::uint8_t> mask(n,1),enabled(g,1);mask[2]=mask[8]=0;enabled[3]=0;
    auto hm=anndata::Array::borrow_dense(mask.data(),{n},{1});auto hg=anndata::Array::borrow_dense(enabled.data(),{g},{1});
    auto dm=ac::DeviceBuffer(n,stream.device),dg=ac::DeviceBuffer(g,stream.device);
    ac::copy_to_device(dm.span(),0,mask.data(),n,stream);ac::copy_to_device(dg.span(),0,enabled.data(),g,stream);
    auto ds=upload(hs.view(),stream),du=upload(hu.view(),stream),dw=upload(hw.view(),stream);
    auto genes=anndata::Axis::gather({4,0,2,3,1});
    velocity::PreparedInput cpu({hs.view(),hu.view(),hw.view(),genes,hm.view(),hg.view()});
    vc::Input input{dw.view,ds.view,du.view,genes};input.fit_mask=ac::DenseView{dm.span(),0,n,1,1,1,ScalarType::uint8};
    input.gene_mask=ac::DenseView{dg.span(),0,g,1,1,1,ScalarType::uint8};vc::PreparedInput gpu(input,stream);
    auto precision=std::is_same_v<T,double>?velocity::ComputePrecision::float64:velocity::ComputePrecision::float32;
    for(bool first:{false,true})for(bool second:{false,true}) {
        velocity::StochasticRunOptions options;options.compute_precision=precision;options.block_genes=2;options.unspliced_second=true;
        velocity::StochasticWorkspace cw(cpu,options);vc::Workspace gw(gpu,options);
        auto cv=anndata::Array::empty<double>({n,g}),cr=anndata::Array::empty<double>({n,g});
        Output gv(n,g,ScalarType::float64,stream,true),gr(n,g,ScalarType::float64,stream),gm(n,g,ScalarType::float64,stream);
        velocity::StochasticOptions fit;fit.initialization.fit_intercept=first;fit.fit_second_intercept=second;
        fit.initialization.min_r2=-100;fit.initialization.min_ratio=-100;
        auto c=velocity::reference::stochastic_into(cpu,cw,cv.edit(),fit,cr.edit());
        auto d=vc::reference::stochastic(gpu,gw,fit,{.velocity=gv.view,.second_residual=gr.view,.spliced=gm.view},stream);
        double tolerance=precision==velocity::ComputePrecision::float32?2e-4:1e-9;
        close(read(gv.view,stream),host(cv.view()),tolerance,"stochastic velocity parity");
        close(read(gr.view,stream),host(cr.view()),tolerance,"second residual parity");
        auto result=d.view();close(read(result.gamma,stream),host(c.gamma.view()),tolerance,"joint gamma parity");
        close(read(result.offset,stream),host(c.offset.view()),tolerance,"joint offset parity");
        close(read(result.offset2,stream),host(c.offset2.view()),tolerance,"second offset parity");
        close(read(result.first_scale,stream),host(c.first_scale.view()),tolerance,"first scale parity");
        close(read(result.second_scale,stream),host(c.second_scale.view()),tolerance,"second scale parity");
        close(read(result.objective,stream),host(c.objective.view()),precision==velocity::ComputePrecision::float32?5e-4:1e-9,"objective parity");
        // Integer/status conversion uses an explicitly representable device cast.
        close(read(result.status,stream),host(c.status.view()),0,"status parity");
        close(read(result.selected,stream),host(c.initialization.genes.view()),0,"selection parity");
        expect(gw.scratch_bytes()<=options.workspace_bytes,"bounded workspace");
    }
    velocity::RunOptions options;options.compute_precision=precision;options.block_genes=2;
    velocity::Workspace cw(cpu,options);vc::Workspace gw(gpu,options);auto cv=anndata::Array::empty<double>({n,g});Output gv(n,g,ScalarType::float64,stream);
    velocity::FitOptions fit;fit.min_r2=100;fit.low_signal=velocity::LowSignal::scvelo_percentile;
    auto c=velocity::reference::steady_state_into(cpu,cw,cv.edit(),fit);auto d=vc::reference::steady_state(gpu,gw,fit,{.velocity=gv.view},stream);
    close(read(d.view().selected,stream),host(c.genes.view()),0,"low-signal selection parity");
    close(read(gv.view,stream),host(cv.view()),precision==velocity::ComputePrecision::float32?2e-4:1e-9,"deterministic velocity parity");
    invalid([&]{vc::Workspace small(gpu,velocity::RunOptions{.workspace_bytes=1});},"insufficient workspace rejected");
    auto aliases=gv.view;aliases.row_stride=0;
    invalid([&]{vc::ReferenceStages stages(gpu,gw,d.view(),fit,{.velocity=aliases},stream);},"overlapping output rejected");
}
void neighborhood(ac::Stream stream) {
    std::vector<double> values{0,2,-3,.2,.3,.7};std::vector<std::uint8_t> indices{0,1,1,2,2,0};std::vector<std::uint64_t> offsets{0,3,6,6};
    auto cpu=anndata::Array::borrow_sparse(anndata::Layout::csr,3,3,values.size(),values.data(),indices.data(),offsets.data());
    auto source=upload(cpu.view(),stream);
    for(auto type:{ScalarType::float32,ScalarType::float64}) {
        Output result(3,3,type,stream);vc::reference::prepare_neighborhood(source.view,result.view,stream);
        double w=type==ScalarType::float32?double(float(1./3)):1./3;
        close(read(result.view,stream),{1,0,0,w,w,w,0,0,1},0,"neighborhood duplicate support and FP32/FP64 normalization");
    }
}
void duplicate_cancellation(ac::Stream stream) {
    std::vector<float> weights{.1234567f},values{1e7f,-9999999.f},u{2};
    std::vector<std::uint8_t> indices{0,0};std::vector<std::uint64_t> offsets{0,2};
    auto hw=anndata::Array::borrow_dense(weights.data(),{1,1},{1,1});
    auto hs=anndata::Array::borrow_sparse(anndata::Layout::csr,1,1,values.size(),values.data(),indices.data(),offsets.data());
    auto hu=anndata::Array::borrow_dense(u.data(),{1,1},{1,1});
    velocity::PreparedInput cpu({hs.view(),hu.view(),hw.view()});float expected_s=0,expected_u=0;
    velocity::reference::moments(cpu,0,1,&expected_s,&expected_u);
    auto dw=upload(hw.view(),stream),ds=upload(hs.view(),stream),du=upload(hu.view(),stream);
    vc::PreparedInput gpu({dw.view,ds.view,du.view},stream);velocity::RunOptions options;options.compute_precision=velocity::ComputePrecision::float32;
    vc::Workspace workspace(gpu,options);vc::FitResult result(gpu,workspace.compute_precision());
    vc::ReferenceStages stages(gpu,workspace,result.view(),velocity::FitOptions{},{},stream);auto moments=stages.moments(0,1);
    close(read(moments.spliced,stream),{expected_s},2e-6,"coalesce duplicate source before weight multiplication");
    close(read(moments.unspliced,stream),{expected_u},2e-6,"duplicate-case unspliced moment");
    expect(std::abs(expected_s-weights[0])<1e-7,"CPU duplicate contract uses logical source sum");
    stages.emit(moments,stages.fit(moments));stages.finish();ac::synchronize(stream);
}
void staging_budget(ac::Stream stream) {
    constexpr std::size_t n=12,g=8;std::vector<double> weights(n*n);for(std::size_t i=0;i<n;++i)weights[i*n+i]=1;
    auto host_weights=dense<float>(n,n,weights);auto dw=upload(host_weights.view(),stream);int calls=0;
    vc::Input input;input.weights=dw.view;input.source_genes=g;input.source_spliced_dtype=input.source_unspliced_dtype=ScalarType::float16;
    input.source={&calls,[](void* p,const std::uint64_t*,std::size_t,ac::Stream,vc::SourceBlock&) {++*static_cast<int*>(p);},500};
    vc::PreparedInput gpu(input,stream);velocity::RunOptions options;options.workspace_bytes=512;
    invalid([&]{vc::Workspace workspace(gpu,options);},"base provider staging charged before block selection");
    expect(calls==0,"workspace construction never acquires source blocks");
    input.source.staging_bytes=0;vc::PreparedInput borrowed(input,stream);vc::Workspace workspace(borrowed,options);
    expect(workspace.compute_precision()==velocity::ComputePrecision::float32,"provider half metadata selects FP32");
    expect(workspace.block_genes()==2 && workspace.scratch_bytes()+workspace.staging_bytes()<=workspace.budget_bytes(),"bounded provider batch size");
}
}
int main() {
    try {
        int device=0;runtime(cudaSetDevice(device));cudaStream_t raw;runtime(cudaStreamCreateWithFlags(&raw,cudaStreamNonBlocking));ac::Stream stream{raw,device};
        compare_case<anndata::Float16>(stream,false);compare_case<float>(stream,false);compare_case<double>(stream,false);
        compare_case<anndata::Float16>(stream,true);compare_case<float>(stream,true);compare_case<double>(stream,true);
        neighborhood(stream);duplicate_cancellation(stream);staging_budget(stream);ac::synchronize(stream);runtime(cudaStreamDestroy(raw));
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 2;}
    if(failures)return 1;std::cout<<"CUDA reference parity and contracts passed\n";
}
