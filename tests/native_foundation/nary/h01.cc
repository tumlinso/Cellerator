#include <Cellerator/compute/operation/indexed_mechanism/incidence.hh>
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
namespace ix=cellerator::compute::operation::indexed;namespace ex=cellerator::execution;
void check(bool b,const char* s){if(!b)throw std::runtime_error(s);}
ex::persistent_axis_identity axis(unsigned id) {
    return {{ex::biological_abi_version,ex::serialized_record_kind::persistent_axis_identity,sizeof(ex::persistent_axis_identity)},
        {id,1},{id,2},{id,3},{id,4}};
}
int main()try {
    ix::indexed_axis input{axis(1),3},output{axis(2),2};
    std::array<ix::argument_index,3> arguments{{{0,{10,1},0,0},{1,{20,1},0,1},{2,{10,1},0,0}}};
    std::array<ix::output_index,2> outputs{{{0,{30,1},0,0,{50,1}},{1,{40,1},0,1,{50,1}}}};
    ix::mechanism_incidence mechanism{{100,1},{200,1},arguments,outputs};
    ix::argument_incidence incidence;
    check(incidence.prepare({&input,1},{&output,1},{&mechanism,1})==ix::incidence_status::success,"prepare ordered repeated arguments");
    std::array<double,3> values{10,2,7},gathered{};ix::host_axis_f64 binding{input.identity,values};
    check(incidence.gather_f64(0,{&binding,1},gathered)==ix::incidence_status::success,"real host gather");
    check(gathered==std::array<double,3>{10,2,10} && gathered[0]-gathered[1]*gathered[2]==-10,"noncommutative independent oracle");
    std::sort(arguments.begin(),arguments.end(),[](auto a,auto b){return a.index<b.index;});
    std::reverse(outputs.begin(),outputs.end());
    check(incidence.prepare({&input,1},{&output,1},{&mechanism,1})==ix::incidence_status::success,"physical reorder accepted");
    check(incidence.gather_f64(0,{&binding,1},gathered)==ix::incidence_status::success && gathered==std::array<double,3>{10,2,10},"sorting does not impose commutativity");
    const auto& saved=incidence.mechanisms()[0];
    check(saved.arguments[0].role.low==10 && saved.arguments[1].role.low==20 && saved.arguments[2].role.low==10 && saved.outputs[0].role.low==30,"roles and output slots retained");
    arguments[0].index=2; // Prepared topology owns its copy.
    check(incidence.gather_f64(0,{&binding,1},gathered)==ix::incidence_status::success && gathered[0]==10,"source record lifetime independent");
    arguments[0].slot=arguments[1].slot;
    check(incidence.prepare({&input,1},{&output,1},{&mechanism,1})==ix::incidence_status::invalid_slot,"duplicate semantic slot rejected");
    check(incidence.gather_f64(0,{&binding,1},gathered)==ix::incidence_status::success && gathered[0]==10,"failed prepare retains prior topology");
    auto wrong=binding;wrong.identity.order.low+=1;gathered.fill(-999);
    check(incidence.gather_f64(0,{&wrong,1},gathered)==ix::incidence_status::invalid_binding && gathered[0]==-999 && gathered[2]==-999,"wrong order rejects without writes");
    check(incidence.gather_f64(0,{&binding,1},values)==ix::incidence_status::invalid_binding && values[0]==10,"gather alias rejected");
    ix::output_index destination{0,{30,1},0,0,{50,1}};
    ix::mechanism_incidence writers[2]{{{101,1},{200,1},{},{&destination,1}},{{102,1},{200,1},{},{&destination,1}}};
    check(incidence.prepare({&input,1},{&output,1},writers)==ix::incidence_status::duplicate_writer,"duplicate overwrite rejected");
    destination.effect.update=ex::output_update_kind::accumulate;destination.effect.requires_initialized_destination=true;
    check(incidence.prepare({&input,1},{&output,1},writers)==ix::incidence_status::success,"explicit shared assembly accepted");
    destination.effect.requires_initialized_destination=false;
    check(incidence.prepare({&input,1},{&output,1},writers)==ix::incidence_status::invalid_effect,"uninitialized accumulation contract rejected");
    std::cout<<"H01 ordered repeated argument gather, physical reorder, ownership and no-write failure controls passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
