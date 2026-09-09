#include "fixture.cuh"
#include <Cellerator/execution/native_value_instance/atom_binding.hh>
namespace ap=ex::atom_plane;
namespace vp=ex::projection_value_plane;
void atom_test(){
    storage owner;cuda_ok(cudaStreamCreate(&owner.a));
    rel::operation_descriptor f;f.topology={{40,1},{1},axis(1),axis(2),{41,1},3};f.dense_width=16;
    auto t=f;t.direction=rel::orientation::transpose;
    {std::vector<std::uint32_t> offsets{0,2,3},sources{0,1,0};
     ok(rel::prepare_relation_pair(f,t,{offsets.data(),3,sources.data(),3},{0,0},owner.a,&owner.first));}
    cuda_ok(cudaMalloc(&owner.weights,6));cuda_ok(cudaMalloc(&owner.input,32*sizeof(float)));cuda_ok(cudaMalloc(&owner.output,32*sizeof(float)));
    __half* alternate{};float *gradient{},*scratch{};
    cuda_ok(cudaMalloc(&alternate,6));cuda_ok(cudaMalloc(&gradient,12));cuda_ok(cudaMalloc(&scratch,12));
    try {
        ex::relation_structure structure{{7,1},{1},{{1,1},{1,1},{1,1},{1,1}},{{2,1},{2,1},{2,1},{2,1}},{1,1},3};
        rel::native_atom_association association{f,structure.identity,structure.source_axis,structure.destination_axis};
        std::uint64_t descriptor=1;
        ap::structural_atom_plane_binding_v1 structural{};structural.descriptor_alignment=alignof(decltype(descriptor));
        structural.plane_identity={1,1};structural.persistent_order_identity={1,2};structural.structure=&structure;
        structural.structure_identity=structure.identity;structural.structure_epoch_value=structure.epoch;
        structural.source_order=structure.source_axis.order;structural.destination_order=structure.destination_axis.order;
        structural.logical_edge_order=f.topology.logical_edge_order;structural.source_descriptor=&descriptor;
        structural.source_descriptor_bytes=sizeof(descriptor);structural.logical_edge_count=3;
        std::array<ex::u64,3> map{0,1,2};
        vp::projection_value_component_v1 component{};component.component_identity=1;
        component.kind=vp::value_component_kind_v1::logical;component.flags=vp::component_trainable_v1|vp::component_gradient_bound_v1;
        component.physical_order=f.topology.logical_edge_order;component.values=owner.weights;component.gradients=gradient;
        component.slot_to_logical_edge=map.data();component.location={ex::residency_kind::device,{},0,0};
        component.slot_count=3;component.value_bytes=6;component.gradient_bytes=12;
        vp::projection_value_plane_v1 plane{};plane.structure=structure.identity;plane.structure_epoch_value=structure.epoch;
        plane.generation={1};plane.logical_edge_order=f.topology.logical_edge_order;
        plane.numeric={ex::numeric_type::f16,ex::numeric_type::f32,ex::numeric_type::f32,0};plane.quantization.kind=ex::quantization_kind::none;
        plane.components=&component;plane.component_count=plane.required_component_count=1;plane.logical_edge_count=3;
        ap::relation_value_atom_plane_v1 atom{};atom.plane_identity={2,1};atom.structural_plane_identity=structural.plane_identity;
        atom.structural_binding=&structural;atom.values=&plane;atom.expected_generation={1};
        std::array<__half,3> weights{__float2half(2),__float2half(-1),__float2half(3)};
        cuda_ok(cudaMemcpy(owner.weights,weights.data(),6,cudaMemcpyHostToDevice));
        ok(rel::publish_atom_values(*owner.first,atom,association,owner.a));cuda_ok(cudaStreamSynchronize(owner.a));
        rel::preparation_report before{};ok(rel::inspect(*owner.first,&before));
        component.values=alternate;plane.generation=atom.expected_generation={2};
        weights={__float2half(4),__float2half(-2),__float2half(6)};
        cuda_ok(cudaMemcpy(alternate,weights.data(),6,cudaMemcpyHostToDevice));
        map[1]=vp::permanent_hole_logical_edge_v1;
        require(!rel::publish_atom_values(*owner.first,atom,association,owner.a),"padding hole cannot become parameter");
        map[1]=0;require(!rel::publish_atom_values(*owner.first,atom,association,owner.a),"duplicate logical owner rejected");map[1]=1;
        auto wrong=association;wrong.native_relation.topology.identity={99,1};
        require(!rel::publish_atom_values(*owner.first,atom,wrong,owner.a),"wrong native association rejected");
        structure.source_axis.domain={99,1};
        require(!rel::publish_atom_values(*owner.first,atom,association,owner.a),"foreign source domain rejected");
        structure.source_axis=association.atom_source;
        structure.destination_axis.order={99,1};structural.destination_order=structure.destination_axis.order;
        require(!rel::publish_atom_values(*owner.first,atom,association,owner.a),"foreign destination order rejected");
        structure.destination_axis=association.atom_destination;structural.destination_order=structure.destination_axis.order;
        rel::preparation_report rejected{};ok(rel::inspect(*owner.first,&rejected));
        require(rejected.value_refreshes==1 && rejected.latest_enqueued_generation.value==1,"rejections submit no publication");
        ok(rel::publish_atom_values(*owner.first,atom,association,owner.a));
        std::vector<float> x(32);for(int i=0;i<32;++i)x[i]=float(i+1)/8;
        cuda_ok(cudaMemcpy(owner.input,x.data(),128,cudaMemcpyHostToDevice));
        ok(rel::enqueue(*owner.first,f,{owner.input,32,rel::input_axis(f),0},{owner.output,32,rel::result_axis(f),0},{2},owner.a));
        cuda_ok(cudaStreamSynchronize(owner.a));std::vector<float> y(32);cuda_ok(cudaMemcpy(y.data(),owner.output,128,cudaMemcpyDeviceToHost));
        for(int c=0;c<16;++c)require(std::abs(y[c]-(4*x[c]-2*x[16+c]))<1e-5 && std::abs(y[16+c]-6*x[c])<1e-5,"new-address generation executes");
        rel::preparation_report after{};ok(rel::inspect(*owner.first,&after));
        require(before.structural_preparation_id==after.structural_preparation_id && after.value_refreshes==2,"rebind preserves counted topology");
        rel::relation_calculus_descriptor calculus;calculus.forward=f;calculus.transpose=t;
        ok(rel::prepare_relation_gradient(*owner.first,calculus,{},owner.a));
        vp::direct_gradient_component_v1 gc{};gc.component_identity=1;gc.physical_order=component.physical_order;
        gc.gradients=gradient;gc.slot_to_logical_edge=map.data();gc.slot_count=3;gc.gradient_bytes=12;
        ap::gradient_atom_plane_v1 ga{};ga.plane_identity={3,1};ga.primal=&atom;ga.primal_generation={2};ga.gradient_generation={1};ga.components=&gc;ga.component_count=1;
        rel::edge_layout_view layout;ok(rel::inspect_edge_layout(*owner.first,&layout));
        rel::edge_plane_view physical{scratch,3,f.topology.identity,f.topology.epoch,layout.order,0};rel::gradient_stamp stamp;
        ok(rel::enqueue_atom_gradient(*owner.first,calculus,{owner.input,32,f.topology.source,0},{owner.input,32,f.topology.destination,0},{1,1},{2,1},ga,association,physical,&stamp,owner.a));
        cuda_ok(cudaStreamSynchronize(owner.a));std::array<float,3> got{};cuda_ok(cudaMemcpy(got.data(),gradient,12,cudaMemcpyDeviceToHost));
        double expected[3]{};for(int c=0;c<16;++c){expected[0]+=double(x[c])*x[c];expected[1]+=double(x[16+c])*x[c];expected[2]+=double(x[c])*x[16+c];}
        for(int e=0;e<3;++e)require(std::abs(got[e]-expected[e])<1e-5,"logical atom gradient matches independent derivative");
        ok(rel::close_relation_pair(&owner.first));
    }catch(...){cudaFree(scratch);cudaFree(gradient);cudaFree(alternate);throw;}
    cudaFree(scratch);cudaFree(gradient);cudaFree(alternate);
}
int main()try{int count=0;cuda_ok(cudaGetDeviceCount(&count));require(count==1,"one leased GPU required");test(1);test(16);atom_test();std::cout<<"V02 real atom rebind, derivative, hole rejection and ownership regressions passed\n";}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
