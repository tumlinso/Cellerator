#include "fixture.cuh"
void epoch_test(){
    storage owner;cuda_ok(cudaStreamCreate(&owner.a));cuda_ok(cudaStreamCreate(&owner.b));
    rel::operation_descriptor f;f.topology={{40,1},{1},axis(1),axis(2),{41,1},3};f.dense_width=16;auto t=f;t.direction=rel::orientation::transpose;
    std::array<std::uint32_t,3> offsets{0,2,3},sources{0,1,0};rel::csr_host_view csr{offsets.data(),3,sources.data(),3};
    ok(rel::prepare_relation_pair(f,t,csr,{0,0},owner.a,&owner.first));ok(rel::create_relation_instance(*owner.first,owner.b,&owner.second));
    cuda_ok(cudaMalloc(&owner.weights,6));cuda_ok(cudaMalloc(&owner.input,128));cuda_ok(cudaMalloc(&owner.output,128));
    std::array<__half,3> values{__float2half(2),__float2half(-1),__float2half(3)};cuda_ok(cudaMemcpy(owner.weights,values.data(),6,cudaMemcpyHostToDevice));
    std::vector<float> input(32,1);cuda_ok(cudaMemcpy(owner.input,input.data(),128,cudaMemcpyHostToDevice));
    rel::device_values_binding binding{owner.weights,3,f.topology.identity,f.topology.epoch,f.topology.logical_edge_order,{1},0};
    ok(rel::publish_values(*owner.first,binding,owner.a));ok(rel::publish_values(*owner.second,binding,owner.b));
    rel::relation_calculus_descriptor calculus;calculus.forward=f;calculus.transpose=t;
    ok(rel::prepare_relation_gradient(*owner.first,calculus,{},owner.a));
    rel::edge_layout_view old_layout{};ok(rel::inspect_edge_layout(*owner.first,&old_layout));
    rel::edge_plane_view old_gradient{owner.output,3,f.topology.identity,f.topology.epoch,old_layout.order,0};
    rel::gradient_stamp old_stamp{};
    ok(rel::enqueue_edge_gradient(*owner.first,calculus,{owner.input,32,f.topology.source,0},{owner.input,32,f.topology.destination,0},{1,1},{2,1},{1},old_gradient,&old_stamp,owner.a));
    rel::preparation_report before{};ok(rel::inspect(*owner.first,&before));
    rel::value_read_lease lease{};ok(rel::begin_value_read(*owner.first,{1},owner.b,&lease));auto stale=lease;
    auto next=f;next.topology.epoch={2};auto next_t=next;next_t.direction=rel::orientation::transpose;
    require(!rel::replace_relation_epoch(&owner.first,next,next_t,csr,{0,0},owner.a),"live borrow prevents replacement");
    rel::preparation_report unchanged{};ok(rel::inspect(*owner.first,&unchanged));require(unchanged.structural_preparation_id==before.structural_preparation_id,"rejected replacement rebuilds nothing");
    ok(rel::end_value_read(*owner.first,lease,owner.b));
    require(!rel::replace_relation_epoch(&owner.first,f,t,csr,{0,0},owner.a),"same epoch replacement rejected");
    auto bad=csr;bad.source_index_count=1;require(!rel::replace_relation_epoch(&owner.first,next,next_t,bad,{0,0},owner.a),"failed new preparation preserves old instance");
    // One old read remains queued; successful replacement drains before freeing.
    ok(rel::enqueue(*owner.first,f,{owner.input,32,rel::input_axis(f),0},{owner.output,32,rel::result_axis(f),0},{1},owner.a));
    ok(rel::replace_relation_epoch(&owner.first,next,next_t,csr,{0,0},owner.a));
    rel::preparation_report replaced{},sibling{};ok(rel::inspect(*owner.first,&replaced));ok(rel::inspect(*owner.second,&sibling));
    require(replaced.structural_preparation_id!=before.structural_preparation_id && sibling.structural_preparation_id==before.structural_preparation_id,"only changed relation gets new preparation");
    require(sibling.epoch.value==1 && sibling.latest_enqueued_generation.value==1 && sibling.structural_instance_count==1,"unaffected sibling retains old epoch");
    require(replaced.epoch.value==2 && replaced.latest_enqueued_generation.value==0,"new epoch has no inherited values");
    require(!rel::publish_values(*owner.first,binding,owner.a),"stale epoch values rejected");
    binding.epoch={2};ok(rel::publish_values(*owner.first,binding,owner.a));
    rel::value_update_request stale_gradient;stale_gradient.kind=rel::value_update_kind::gradient_step;
    stale_gradient.operand=old_gradient;stale_gradient.expected={1};stale_gradient.next={2};stale_gradient.alpha=1;stale_gradient.gradient=old_stamp;
    require(!rel::enqueue_value_update(*owner.first,stale_gradient,owner.a),"old epoch gradient response rejected");
    rel::value_read_lease fresh{};ok(rel::begin_value_read(*owner.first,{1},owner.b,&fresh));
    require(!rel::end_value_read(*owner.first,stale,owner.b),"old epoch ticket cannot consume new lease");ok(rel::end_value_read(*owner.first,fresh,owner.b));
    ok(rel::close_relation_pair(&owner.first));ok(rel::close_relation_pair(&owner.second));
    // Actual allocator address reuse with identical semantic metadata must still
    // reject a ticket from the retired pair's distinct incarnation.
    std::uintptr_t prior_address=0;rel::value_read_lease prior_ticket{};bool reused=false;
    for(int attempt=0;attempt<32 && !reused;++attempt){
        ok(rel::prepare_relation_pair(f,t,csr,{0,0},owner.a,&owner.first));binding.epoch={1};ok(rel::publish_values(*owner.first,binding,owner.a));
        rel::value_read_lease current{};ok(rel::begin_value_read(*owner.first,{1},owner.b,&current));
        if(prior_address){auto copy=prior_ticket;require(!rel::end_value_read(*owner.first,copy,owner.b),"retired incarnation cannot return current borrow");}
        reused=prior_address==reinterpret_cast<std::uintptr_t>(owner.first);
        prior_address=reinterpret_cast<std::uintptr_t>(owner.first);prior_ticket=current;
        ok(rel::end_value_read(*owner.first,current,owner.b));ok(rel::close_relation_pair(&owner.first));
    }
    require(reused,"actual pair allocation address reuse exercised");
}
int main()try{int count=0;cuda_ok(cudaGetDeviceCount(&count));require(count==1,"one leased GPU required");epoch_test();std::cout<<"V05 safe epoch replacement, unaffected sibling and actual address-reuse stale-ticket rejection passed\n";}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
