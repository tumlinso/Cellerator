#pragma once
#include "async_fixture.cuh"
void f32_test(int width,bool derive){
    storage owner;cuda_ok(cudaStreamCreate(&owner.a));cuda_ok(cudaStreamCreate(&owner.b));
    rel::operation_descriptor f;f.topology={{40,1},{1},axis(1),axis(2),{41,1},3};f.dense_width=width;f.arithmetic.relation_storage=ex::numeric_type::f32;
    auto t=f;t.direction=rel::orientation::transpose;
    std::array<std::uint32_t,3> offsets{0,2,3},sources{0,1,0};
    ok(rel::prepare_relation_pair(f,t,{offsets.data(),3,sources.data(),3},{0,0,derive},owner.a,&owner.first));
    ok(rel::create_relation_instance(*owner.first,owner.b,&owner.second));
    float* values{};cuda_ok(cudaMalloc(&values,12));cuda_ok(cudaMalloc(&owner.input,2*width*4));cuda_ok(cudaMalloc(&owner.output,2*width*4));
    try{
        std::vector<float> x(2*width,0);for(int c=0;c<width;++c)x[c]=1;
        cuda_ok(cudaMemcpy(owner.input,x.data(),x.size()*4,cudaMemcpyHostToDevice));
        auto check=[&](rel::prepared_relation_pair& pair,cudaStream_t stream,const rel::operation_descriptor& op,float first,std::uint64_t gen,bool projected){
            rel::device_state_view in{owner.input,std::uint64_t(x.size()),rel::input_axis(op),0};rel::device_result_view out{owner.output,std::uint64_t(x.size()),rel::result_axis(op),0};
            if(projected)ok(rel::enqueue_derived_f16(pair,op,in,out,{gen},stream));else ok(rel::enqueue(pair,op,in,out,{gen},stream));
            cuda_ok(cudaStreamSynchronize(stream));std::vector<float> got(x.size());cuda_ok(cudaMemcpy(got.data(),owner.output,got.size()*4,cudaMemcpyDeviceToHost));
            for(int c=0;c<width;++c)require(got[c]==first && got[width+c]==(op.direction==rel::orientation::forward?3.f:2.f),"f32 values survive exact observable execution");
        };
        for(std::uint64_t gen=1;gen<=2;++gen){
            const float changed=1.f+std::ldexp(1.f,int(gen)-19);
            std::array<float,3> weights{changed,2,3};cuda_ok(cudaMemcpy(values,weights.data(),12,cudaMemcpyHostToDevice));
            rel::device_f32_values_binding binding{values,3,f.topology.identity,f.topology.epoch,f.topology.logical_edge_order,{gen},0};
            ok(rel::publish_f32_values(*owner.first,binding,owner.a));
            if(gen==1){ok(rel::publish_f32_values(*owner.second,binding,owner.b));cuda_ok(cudaStreamSynchronize(owner.b));}
            check(*owner.first,owner.a,f,changed,gen,false);check(*owner.first,owner.a,t,changed,gen,false);
            rel::preparation_report report{};ok(rel::inspect(*owner.first,&report));
            require(report.instance_value_bytes==std::uint64_t(derive?18:12),"only explicitly requested derived value allocation");
            require(report.derived_f16_generation.value==(derive?gen:0),"derived generation tracks authoritative publication");
            if(derive){check(*owner.first,owner.a,f,1.f,gen,true);check(*owner.first,owner.a,t,1.f,gen,true);}
            else require(!rel::enqueue_derived_f16(*owner.first,f,{owner.input,std::uint64_t(x.size()),rel::input_axis(f),0},{owner.output,std::uint64_t(x.size()),rel::result_axis(f),0},{gen},owner.a),"unrequested low precision route rejected");
        }
        ok(rel::close_relation_pair(&owner.first));check(*owner.second,owner.b,f,1.f+std::ldexp(1.f,-18),1,false);
        ok(rel::close_relation_pair(&owner.second));
    }catch(...){cudaFree(values);throw;}cudaFree(values);
}
