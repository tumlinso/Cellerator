// Physical-order sparse gradients compared with independent logical N16 loops.
#include "../../src/compute/operation/prepared_relation.cu"
#include "../semantic_spine/native/test_require.hh"
#include <cuda_fp16.h>
#include <cmath>
#include <iostream>
namespace ce = cellerator::compute::relation;
namespace ex = cellerator::execution;
static void gpu(cudaError_t s) { SPINE_REQUIRE(s == cudaSuccess); }
static void check(ce::status s) {
    if (!s) std::cerr << s.message << '\n';
    SPINE_REQUIRE(s);
}
static ce::axis_descriptor axis(unsigned seed, unsigned count) {
    ce::axis_descriptor a{}; a.extent=count;
    a.identity.header={1,ex::serialized_record_kind::persistent_axis_identity,sizeof(a.identity)};
    a.identity.domain={seed,99}; a.identity.order={seed+1,88};
    a.identity.geometry={seed+2,77}; a.identity.partition={seed+3,66}; return a;
}
static void fixture(unsigned rows, unsigned cols, unsigned width,
                    const std::vector<unsigned>& offsets, const std::vector<unsigned>& sources, bool half_rounded) {
    ce::operation_descriptor f{}; f.dense_width=width;
    f.topology.identity={101,202}; f.topology.epoch={5}; f.topology.logical_edge_order={303,404};
    f.topology.source=axis(10,cols); f.topology.destination=axis(20,rows); f.topology.edge_count=sources.size();
    auto t=f; t.direction=ce::orientation::transpose;
    cudaStream_t stream{}; gpu(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    ce::prepared_relation_pair* pair=nullptr;
    check(ce::prepare_relation_pair(f,t,{offsets.data(),offsets.size(),sources.data(),sources.size()},{0,1<<24},stream,&pair));
    ce::relation_calculus_descriptor calculus{};calculus.forward=f;calculus.transpose=t;
    calculus.gradient=half_rounded?ce::gradient_arithmetic::round_operands_f16_rne:ce::gradient_arithmetic::full_f32;
    check(ce::prepare_relation_gradient(*pair,calculus,{ce::gradient_route::force_sparse,1<<24},stream));
    SPINE_REQUIRE(!ce::prepare_relation_gradient(*pair,calculus,{},stream));
    ce::edge_layout_view layout{};check(ce::inspect_edge_layout(*pair,&layout));
    std::vector<bool> seen(sources.size());
    for(unsigned e=0;e<sources.size();++e) {
        auto physical=layout.logical_to_physical[e];SPINE_REQUIRE(physical<sources.size()&&!seen[physical]);
        seen[physical]=true;SPINE_REQUIRE(pair->physical_to_logical[physical]==e);
    }
    const auto projection=pair->report.forward_projection;
    std::vector<__half> weights(sources.size());
    __half* dw=nullptr; if (!weights.empty()) gpu(cudaMalloc(&dw,weights.size()*2));
    for (unsigned generation=1;generation<=2;++generation) {
        for (unsigned e=0;e<weights.size();++e) weights[e]=__float2half(float(int(e%7)-3)*generation/2);
        if(dw) gpu(cudaMemcpyAsync(dw,weights.data(),weights.size()*2,cudaMemcpyHostToDevice,stream));
        check(ce::publish_values(*pair,{dw,weights.size(),f.topology.identity,f.topology.epoch,f.topology.logical_edge_order,{generation},0},stream));
        std::vector<float> x(cols*width), y(rows*width,123);
        std::vector<double> reference(rows*width,0);
        for(unsigned i=0;i<x.size();++i) x[i]=float(int((i+generation)%11)-5)/8;
        for(unsigned row=0;row<rows;++row) for(unsigned e=offsets[row];e<offsets[row+1];++e)
            for(unsigned k=0;k<width;++k) reference[row*width+k]+=double(__half2float(weights[e]))*x[sources[e]*width+k];
        float *dx=nullptr,*dy=nullptr;
        if(!x.empty()) {gpu(cudaMalloc(&dx,x.size()*4));gpu(cudaMemcpyAsync(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice,stream));}
        if(!y.empty()) {gpu(cudaMalloc(&dy,y.size()*4));gpu(cudaMemcpyAsync(dy,y.data(),y.size()*4,cudaMemcpyHostToDevice,stream));}
        ce::device_state_view input{dx,x.size(),f.topology.source,0};
        ce::device_result_view output{dy,y.size(),f.topology.destination,0};
        auto reject=[&](const ce::operation_descriptor& op,const ce::device_state_view& in,const ce::device_result_view& out) {
            const auto before=pair->report.accepted_forward_launches;
            SPINE_REQUIRE(!ce::enqueue(*pair,op,in,out,{generation},stream));
            SPINE_REQUIRE(pair->report.accepted_forward_launches==before);
            gpu(cudaStreamSynchronize(stream));
            if(dy) {gpu(cudaMemcpy(y.data(),dy,y.size()*4,cudaMemcpyDeviceToHost));for(auto v:y) SPINE_REQUIRE(v==123);}
        };
        if(!x.empty()) {auto short_input=input;--short_input.count;reject(f,short_input,output);}
        if(!y.empty()) {auto short_output=output;--short_output.count;reject(f,input,short_output);}
        auto bad_axis=input; ++bad_axis.axis.identity.order.high; reject(f,bad_axis,output);
        auto bad_width=f; bad_width.dense_width=7; reject(bad_width,input,output);
        auto overflow=f; overflow.topology.source.extent=~std::uint64_t(0); reject(overflow,input,output);
        if(dx && dy && x.size()>1) {auto alias=output;alias.data=dx+1;reject(f,input,alias);}
        check(ce::enqueue(*pair,f,input,output,{generation},stream)); gpu(cudaStreamSynchronize(stream));
        if(dy) gpu(cudaMemcpy(y.data(),dy,y.size()*4,cudaMemcpyDeviceToHost));
        for(unsigned i=0;i<y.size();++i) SPINE_REQUIRE(std::isfinite(y[i]) && std::abs(y[i]-reference[i])<=2e-6+2e-6*std::abs(reference[i]));
        std::vector<float> cotangent(rows*16), gradient(sources.size(), 123);
        for(unsigned i=0;i<cotangent.size();++i)cotangent[i]=float(int(i%19)-9)/7.3f;
        float *dc=nullptr,*dg=nullptr;
        if(!cotangent.empty()){gpu(cudaMalloc(&dc,cotangent.size()*4));gpu(cudaMemcpyAsync(dc,cotangent.data(),cotangent.size()*4,cudaMemcpyHostToDevice,stream));}
        if(!gradient.empty()){gpu(cudaMalloc(&dg,gradient.size()*4));gpu(cudaMemcpyAsync(dg,gradient.data(),gradient.size()*4,cudaMemcpyHostToDevice,stream));}
        ce::device_state_view cot{dc,cotangent.size(),f.topology.destination,0};
        ce::edge_plane_view gp{dg,gradient.size(),f.topology.identity,f.topology.epoch,layout.order,0};
        ce::gradient_stamp stamp{};stamp.producer_serial=777;
        auto reject_gradient=[&](ce::edge_plane_view target,ce::operand_version version,ex::value_generation gen) {
            auto before=pair->updates.gradient_launches;
            SPINE_REQUIRE(!ce::enqueue_edge_gradient(*pair,calculus,input,cot,version,{22,generation},gen,target,&stamp,stream));
            SPINE_REQUIRE(stamp.producer_serial==777&&pair->updates.gradient_launches==before);
            gpu(cudaStreamSynchronize(stream));
            if(dg){gpu(cudaMemcpy(gradient.data(),dg,gradient.size()*4,cudaMemcpyDeviceToHost));for(float v:gradient)SPINE_REQUIRE(v==123);}
        };
        reject_gradient(gp,{11,0},{generation});reject_gradient(gp,{11,generation},{generation+1});
        auto wrong=gp;++wrong.order.high;reject_gradient(wrong,{11,generation},{generation});
        if(!gradient.empty()) {
            wrong=gp;--wrong.count;reject_gradient(wrong,{11,generation},{generation});
            wrong=gp;wrong.f32_data=pair->values;reject_gradient(wrong,{11,generation},{generation});
            wrong=gp;wrong.f32_data=dx;reject_gradient(wrong,{11,generation},{generation});
        }
        check(ce::enqueue_edge_gradient(*pair,calculus,input,cot,{11,generation},{22,generation},{generation},gp,&stamp,stream));
        gpu(cudaStreamSynchronize(stream));
        if(dg)gpu(cudaMemcpy(gradient.data(),dg,gradient.size()*4,cudaMemcpyDeviceToHost));
        for(unsigned row=0;row<rows;++row)for(unsigned e=offsets[row];e<offsets[row+1];++e){
            double sum=0;
            for(unsigned k=0;k<16;++k){
                float a=x[sources[e]*16+k],b=cotangent[row*16+k];
                if(half_rounded){a=__half2float(__float2half_rn(a));b=__half2float(__float2half_rn(b));}
                sum+=double(a)*b;
            }
            auto actual=gradient[layout.logical_to_physical[e]];
            SPINE_REQUIRE(std::isfinite(actual)&&std::abs(actual-sum)<=2e-5+2e-5*std::abs(sum));
        }
        SPINE_REQUIRE(stamp.forward_generation.value==generation&&stamp.input.version==generation);
        SPINE_REQUIRE(stamp.pair_incarnation==pair->incarnation&&stamp.producer_serial==generation);
        SPINE_REQUIRE(pair->last_gradient_output.f32_data==dg);
        SPINE_REQUIRE(pair->updates.gradient_preparations==1&&pair->updates.gradient_launches==generation);
        if(dc)gpu(cudaFree(dc));if(dg)gpu(cudaFree(dg));
        SPINE_REQUIRE(pair->report.topology_preparations==1 && pair->report.accepted_forward_launches==generation);
        SPINE_REQUIRE(ex::same_identity(projection,pair->report.forward_projection));
        if(dx)gpu(cudaFree(dx));if(dy)gpu(cudaFree(dy));
    }
    if(dw)gpu(cudaFree(dw));ce::destroy(pair);gpu(cudaStreamDestroy(stream));
}
int main() {
    gpu(cudaSetDevice(0));cudaDeviceProp p{};gpu(cudaGetDeviceProperties(&p,0));
    SPINE_REQUIRE(p.major==7 && p.minor==0);
    for(bool half_rounded:{false,true}) {
        const unsigned width=16;
        fixture(3,4,width,{0,2,2,4},{3,0,2,1},half_rounded);
        std::vector<unsigned> offsets{0},sources;
        for(unsigned row=0;row<35;++row) {if(row%4) {sources.push_back((row*7+33)%67);sources.push_back((row*7)%67);}offsets.push_back(sources.size());}
        fixture(35,67,width,offsets,sources,half_rounded);
        fixture(3,4,width,{0,0,0,0},{},half_rounded);fixture(0,4,width,{0},{},half_rounded);fixture(0,0,width,{0},{},half_rounded);
    }
    std::cout<<"N16 full-f32 and half-rounded sparse gradient, physical bijection, stamps and rejected bindings PASS\n";
}
