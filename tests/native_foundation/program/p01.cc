#include <Cellerator/execution/program/program_v2.h>
#include <Cellerator/execution/launch_bindings.hh>
#include <Cellerator/execution/program/program_v1_adapter.h>
#include <array>
#include <cstdlib>
#include <iostream>
namespace ce = cellerator::execution;
namespace pg = ce::program;
void require(bool value, int line) {
    if (!value) { std::cerr << "P01 assertion failed at line " << line << '\n'; std::abort(); }
}
#define check(value) require(value, __LINE__)
unsigned calls = 0;
pg::program_status calculate(const void*, const pg::launch_binding_v2& b, void*) noexcept {
    ++calls;
    const auto& t = *b.typed;
    const auto* a = static_cast<const float*>(t.inputs[0].storage.dense.data);
    const auto* x = static_cast<const float*>(t.inputs[1].storage.dense.data);
    const auto* c = static_cast<const float*>(t.inputs[2].storage.dense.data);
    const auto* w = static_cast<const float*>(t.values[0].plane->values);
    auto* y = static_cast<float*>(t.outputs[0].storage.dense.data);
    auto* z = static_cast<float*>(t.outputs[1].storage.dense.data);
    for (unsigned i=0; i<4; ++i) { y[i]=a[i]*x[i]+c[i]*w[i]; z[i]+=a[i]-x[i]; }
    return pg::program_status::success;
}
ce::biological_operand_view dense(float* data, ce::axis_identity axis) {
    ce::biological_operand_view v{};
    v.kind=ce::operand_kind::dense_tensor;
    v.storage.dense.data=data;
    v.storage.dense.location={ce::residency_kind::host,{},-1,0};
    v.storage.dense.value_type=ce::numeric_type::f32;
    v.storage.dense.rank=1; v.storage.dense.axes[0]=axis;
    v.storage.dense.shape[0]=4; v.storage.dense.stride[0]=1;
    return v;
}
pg::program_status legacy_add(const void*, const pg::launch_binding_v2& binding, void*) noexcept {
    *static_cast<float*>(binding.output) = *static_cast<const float*>(binding.input) + 1;
    return pg::program_status::success;
}

int main() {
    // Existing v1 adapter must clear typed contracts even with reused storage.
    float original=3, intermediate=0, result=0;
    pg::legacy_program_entry_v1 entries[]{
        {1,1,nullptr,legacy_add,0,0,0}, {2,1,nullptr,legacy_add,1,0,0}};
    pg::prepared_stage_v2 adapted_stages[2]{};
    pg::prepared_program_v2 adapted{};
    std::uint64_t dependencies[1]{};
    check(pg::adapt_legacy_program_v1({entries,2},adapted_stages,2,dependencies,1,&adapted)==pg::program_status::success);
    pg::launch_binding_v2 legacy_bindings[]{
        {&original,&intermediate,nullptr,nullptr,0},
        {&intermediate,&result,nullptr,nullptr,0}};
    check(pg::execute_prepared_program_v2(adapted,legacy_bindings,2,nullptr)==pg::program_status::success);
    check(intermediate==4 && result==5);

    ce::axis_identity axis{{1,1},{2,1},{3,1},{4,1}};
    ce::relation_structure relation{{100,1},{1},axis,axis,{200,1},4};
    float a[]{1,2,3,4}, x[]{5,6,7,8}, c[]{2,3,4,5}, w[]{2,2,2,2};
    float y[]{99,99,99,99}, z[]{10,10,10,10};
    ce::biological_operand_view inputs[]{dense(a,axis),dense(x,axis),dense(c,axis)};
    ce::biological_operand_view outputs[]{dense(y,axis),dense(z,axis)};
    ce::operand_axis_contract in_contracts[3]{}, out_contracts[2]{};
    for (auto& v:in_contracts) v={ce::operand_kind::dense_tensor,1,{}, {axis}};
    for (auto& v:out_contracts) v={ce::operand_kind::dense_tensor,1,{}, {axis}};
    ce::output_axis_contract orders[2]{};
    for (unsigned i=0;i<2;++i) orders[i]={axis,axis,ce::order_transition_kind::preserve,0,static_cast<ce::u16>(i),1,1,{}, {0,0}};
    ce::output_effect_contract effects[]{
        {ce::output_update_kind::overwrite,false,false,0,ce::invalid_scalar_binding_id,ce::invalid_scalar_binding_id},
        {ce::output_update_kind::accumulate,true,false,0,ce::invalid_scalar_binding_id,ce::invalid_scalar_binding_id}};
    ce::prepared_binding_contract contract{};
    contract.structures[0]={relation.identity,relation.epoch};contract.structure_count=1;
    contract.inputs=in_contracts;contract.input_count=3;
    contract.outputs=out_contracts;contract.output_count=2;
    contract.output_orders=orders;contract.output_order_count=2;
    contract.output_effects=effects;contract.output_effect_count=2;
    contract.workspace={0,1,0};
    ce::value_plane plane{{100,1},{1},w,{ce::residency_kind::host,{},-1,0},
        {ce::numeric_type::f32,ce::numeric_type::f32,ce::numeric_type::f32,0},
        {ce::quantization_kind::none,ce::numeric_type::invalid,ce::numeric_type::invalid,0,nullptr,nullptr,0},
        ce::value_layout_kind::logical_edge_order,{}, {4},4,sizeof(w)};
    ce::value_binding value{&plane,{4}};
    ce::launch_bindings typed{};
    typed.structures=&relation;typed.structure_count=1;
    typed.inputs=inputs;typed.input_count=3;typed.outputs=outputs;typed.output_count=2;
    typed.values=&value;typed.value_count=1;
    typed.stream={nullptr,-1,0};typed.workspace={nullptr,0,{ce::residency_kind::host,{},-1,0}};
    pg::prepared_stage_v2 stage{1,1,nullptr,calculate,0,0,0,0,&contract};
    pg::prepared_program_v2 program{2,0,&stage,1,nullptr,0};
    pg::launch_binding_v2 binding{};binding.typed=&typed;
    auto execute=[&]{return pg::execute_prepared_program_v2(program,&binding,1,nullptr);};
    check(execute()==pg::program_status::success && calls==1);
    for(unsigned i=0;i<4;++i) check(y[i]==a[i]*x[i]+c[i]*w[i] && z[i]==6);
    auto reject=[&]{float before=y[0];check(execute()==pg::program_status::invalid_typed_binding);check(calls==1 && y[0]==before);};
    typed.input_count=2;reject();typed.input_count=3;
    inputs[0].storage.dense.axes[0].order.slot=99;reject();inputs[0].storage.dense.axes[0]=axis;
    value.expected_generation={5};reject();value.expected_generation={4};
    effects[0].requires_initialized_destination=true;reject();effects[0].requires_initialized_destination=false;
    outputs[0].storage.dense.data=a;reject();outputs[0].storage.dense.data=y;
    binding.input=a;reject();binding.input=nullptr;
    binding.typed=nullptr;reject();binding.typed=&typed;
    stage.binding_contract=nullptr;reject();stage.binding_contract=&contract;
    typed.stream.stream=&stage;reject();typed.stream.stream=nullptr;
    typed.stream.device_ordinal=-2;reject();typed.stream.device_ordinal=-1;
    typed.workspace.location.residency=ce::residency_kind::device;reject();
    typed.workspace.location.residency=ce::residency_kind::host;
    typed.stream.device_ordinal=0;reject();typed.stream.device_ordinal=-1;
    // A rejected final stage cannot leave the first stage's accumulated effects.
    pg::prepared_stage_v2 stages[]{stage,stage};
    stages[1].binding_index=1;
    pg::launch_binding_v2 bindings[]{binding,binding};
    ce::launch_bindings bad_typed=typed;
    bad_typed.input_count=2;
    bindings[1].typed=&bad_typed;
    program.stages=stages;program.stage_count=2;
    check(pg::execute_prepared_program_v2(program,bindings,2,nullptr)==pg::program_status::invalid_typed_binding);
    check(calls==1 && z[0]==6);
    stages[1].required_workspace_bytes=8;
    check(pg::execute_prepared_program_v2(program,bindings,2,nullptr)==pg::program_status::insufficient_bindings);
    check(calls==1 && z[0]==6);
    std::cout<<"typed 3-input, 2-output, value generation and mixed effects: passed\n";
}
