#pragma once
#include "fixture.cuh"
void update_test(bool f32,bool derive=false){
    storage owner;cuda_ok(cudaStreamCreate(&owner.a));cuda_ok(cudaStreamCreate(&owner.b));
    rel::operation_descriptor f;f.topology={{40,1},{1},axis(1),axis(2),{41,1},3};f.dense_width=16;if(f32)f.arithmetic.relation_storage=ex::numeric_type::f32;
    auto t=f;t.direction=rel::orientation::transpose;
    std::array<std::uint32_t,3> offsets{0,2,3},sources{0,1,0};ok(rel::prepare_relation_pair(f,t,{offsets.data(),3,sources.data(),3},{0,0,derive},owner.a,&owner.first));ok(rel::create_relation_instance(*owner.first,owner.b,&owner.second));
    float *values{},*cotangent{},*gradient1{},*gradient2{},*delta{};
    cuda_ok(cudaMalloc(&values,12));cuda_ok(cudaMalloc(&owner.weights,6));cuda_ok(cudaMalloc(&owner.input,128));cuda_ok(cudaMalloc(&owner.output,128));
    cuda_ok(cudaMalloc(&cotangent,128));cuda_ok(cudaMalloc(&gradient1,12));cuda_ok(cudaMalloc(&gradient2,12));cuda_ok(cudaMalloc(&delta,12));
    try{
        std::array<float,3> first{1,2,3},second{4,5,6};
        auto publish=[&](rel::prepared_relation_pair& pair,cudaStream_t stream,const std::array<float,3>& w){
            if(f32){cuda_ok(cudaMemcpy(values,w.data(),12,cudaMemcpyHostToDevice));ok(rel::publish_f32_values(pair,{values,3,f.topology.identity,f.topology.epoch,f.topology.logical_edge_order,{1},0},stream));}
            else{std::array<__half,3> half{__float2half(w[0]),__float2half(w[1]),__float2half(w[2])};cuda_ok(cudaMemcpy(owner.weights,half.data(),6,cudaMemcpyHostToDevice));ok(rel::publish_values(pair,{owner.weights,3,f.topology.identity,f.topology.epoch,f.topology.logical_edge_order,{1},0},stream));}
            cuda_ok(cudaStreamSynchronize(stream));
        };publish(*owner.first,owner.a,first);publish(*owner.second,owner.b,second);
        std::vector<float> x(32),dy(32);for(int c=0;c<16;++c){x[c]=1;x[16+c]=2;dy[c]=3;dy[16+c]=4;}
        cuda_ok(cudaMemcpy(owner.input,x.data(),128,cudaMemcpyHostToDevice));cuda_ok(cudaMemcpy(cotangent,dy.data(),128,cudaMemcpyHostToDevice));
        auto check=[&](rel::prepared_relation_pair& pair,cudaStream_t stream,const std::array<float,3>& w,std::uint64_t gen){
            ok(rel::enqueue(pair,f,{owner.input,32,f.topology.source,0},{owner.output,32,f.topology.destination,0},{gen},stream));cuda_ok(cudaStreamSynchronize(stream));
            std::vector<float> got(32);cuda_ok(cudaMemcpy(got.data(),owner.output,128,cudaMemcpyDeviceToHost));
            for(int c=0;c<16;++c)require(got[c]==w[0]+2*w[1] && got[16+c]==w[2],"independent instance update result");
        };
        rel::relation_calculus_descriptor calculus;calculus.forward=f;calculus.transpose=t;
        ok(rel::prepare_relation_gradient(*owner.first,calculus,{},owner.a));ok(rel::prepare_relation_gradient(*owner.second,calculus,{},owner.b));
        rel::edge_layout_view layout{};ok(rel::inspect_edge_layout(*owner.first,&layout));
        auto plane=[&](float* p){return rel::edge_plane_view{p,3,f.topology.identity,f.topology.epoch,layout.order,0};};
        auto response=[&](rel::prepared_relation_pair& pair,cudaStream_t stream,float* output,std::uint64_t gen){
            rel::gradient_stamp stamp;ok(rel::enqueue_edge_gradient(pair,calculus,{owner.input,32,f.topology.source,0},{cotangent,32,f.topology.destination,0},{1,1},{2,1},{gen},plane(output),&stamp,stream));cuda_ok(cudaStreamSynchronize(stream));
            std::array<float,3> got{};cuda_ok(cudaMemcpy(got.data(),output,12,cudaMemcpyDeviceToHost));const std::array<float,3> expected{48,96,64};
            for(int e=0;e<3;++e)require(got[layout.logical_to_physical[e]]==expected[e],"actual response equals independent derivative");return stamp;
        };
        auto old_first=response(*owner.first,owner.a,gradient1,1);auto stable_second=response(*owner.second,owner.b,gradient2,1);
        rel::value_update_request request;request.kind=rel::value_update_kind::gradient_step;request.operand=plane(gradient2);request.expected={1};request.next={2};request.alpha=1.f/16;request.gradient=stable_second;
        require(!rel::enqueue_value_update(*owner.first,request,owner.a),"foreign instance response rejected");
        std::array<float,3> packed{};const float d=f32?std::ldexp(1.f,-18):.25f;packed[layout.logical_to_physical[0]]=d;
        cuda_ok(cudaMemcpy(delta,packed.data(),12,cudaMemcpyHostToDevice));request.kind=rel::value_update_kind::delta_add;request.operand=plane(delta);request.gradient={};
        ok(rel::enqueue_value_update(*owner.first,request,owner.a));first[0]+=d;check(*owner.first,owner.a,first,2);check(*owner.second,owner.b,second,1);
        request.kind=rel::value_update_kind::gradient_step;request.operand=plane(gradient1);request.expected={2};request.next={3};request.gradient=old_first;
        require(!rel::enqueue_value_update(*owner.first,request,owner.a),"parameter update invalidates old response");
        require(!rel::enqueue(*owner.first,f,{owner.input,32,f.topology.source,0},{owner.output,32,f.topology.destination,0},{1},owner.a),"old value generation is not a historical snapshot");
        auto fresh=response(*owner.first,owner.a,gradient1,2);request.gradient=fresh;ok(rel::enqueue_value_update(*owner.first,request,owner.a));
        const std::array<float,3> grad{48,96,64};for(int e=0;e<3;++e)first[e]=std::fma(-1.f/16,grad[e],first[e]);check(*owner.first,owner.a,first,3);
        request.operand=plane(gradient2);request.expected={1};request.next={2};request.gradient=stable_second;
        ok(rel::enqueue_value_update(*owner.second,request,owner.b));for(int e=0;e<3;++e)second[e]=std::fma(-1.f/16,grad[e],second[e]);check(*owner.second,owner.b,second,2);
        rel::preparation_report a{},b{};ok(rel::inspect(*owner.first,&a));ok(rel::inspect(*owner.second,&b));require(a.structural_preparation_id==b.structural_preparation_id,"updates share structure only");
        if(f32)require(a.derived_f16_generation.value==(derive?3:0) && b.derived_f16_generation.value==(derive?2:0),"derived projections follow their own authoritative updates");
        ok(rel::close_relation_pair(&owner.first));ok(rel::close_relation_pair(&owner.second));
    }catch(...){cudaFree(delta);cudaFree(gradient2);cudaFree(gradient1);cudaFree(cotangent);cudaFree(values);throw;}
    cudaFree(delta);cudaFree(gradient2);cudaFree(gradient1);cudaFree(cotangent);cudaFree(values);
}
