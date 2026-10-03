#pragma once
#include <Cellerator/packing/strategies/placement.hh>
#include <Cellerator/compute/operation/native_numeric/local_arithmetic.hh>
#include <array>
#include <cmath>
#include <stdexcept>
namespace pk=cellerator::packing;
namespace st=pk::strategies;
namespace ex=pk::ex;
namespace ix=pk::ix;
namespace rel=pk::rel;
inline int checks=0;
inline void check(bool b) {++checks;if(!b)throw std::runtime_error("strategy check "+std::to_string(checks));}
inline void success(pk::status s) {check(s==pk::status::success);}
inline rel::axis_descriptor axis(std::uint64_t id,std::uint64_t extent) {
    return {{{1,ex::serialized_record_kind::persistent_axis_identity,sizeof(ex::persistent_axis_identity)},
      {id,1},{id,2},{id,3},{id,4}},extent};
}
struct fixture {
    std::array<ix::argument_index,6> arguments{};
    std::array<ix::output_index,6> outputs{};
    std::array<ix::mechanism_incidence,6> jobs{};
    std::array<st::cohort_key,6> keys{};
    std::array<std::uint64_t,6> sources{0,1,3,2,0,3},destinations{0,2,1,2,2,1};
    pk::problem problem{};
    std::array<cellpack::u32,3> groups{0,2,3},blocks{0,2,4};
    pk::cellpack_strategy cellpack{};
    fixture() {
        for(std::size_t i=0;i<6;++i) {
            arguments[i]={0,{11,1},0,sources[i]};outputs[i].role={22,1};outputs[i].index=destinations[i];outputs[i].assembly_owner={33,1};
            outputs[i].effect.update=ex::output_update_kind::accumulate;outputs[i].effect.requires_initialized_destination=true;
            jobs[i]={{100+i,1},{44,1},{&arguments[i],1},{&outputs[i],1}};
        }
        const std::array<std::uint64_t,6> groups_of_parameters{3,1,2,1,3,2};
        for(std::size_t i=0;i<6;++i)keys[i]={{44,1},{groups_of_parameters[i],1}};
        problem.operation.topology={{80,1},{2},axis(1,4),axis(2,3),{90,1},6};
        problem.operation.arithmetic={ex::numeric_type::f64,ex::numeric_type::f64,ex::numeric_type::f64,
            ex::numeric_type::f64,ex::numeric_type::f64,false,false,rel::nonfinite_policy::propagate};
        problem.state=axis(3,3);problem.work=jobs;problem.relation_evaluator={44,1};
        cellpack.plan={3,4,nullptr,nullptr,nullptr,nullptr,2,groups.data(),2,blocks.data()};
    }
    fixture(const fixture&)=delete;
};
struct invocation {
    std::array<double,4> input{0,2,0,4},physical_input{};
    std::array<double,6> weights{2,3,4,5,6,-1};
    std::array<double,3> output{},physical_output{},cotangent{2,-3,4},physical_cotangent{};
    std::array<double,4> gradient{},physical_gradient{};
    void forward(const pk::problem& p,const st::relation_routes& routes) {
        success(pk::convert_rows<double>(input,physical_input,routes.forward.source_order,1,false));
        pk::nn::value_identity values{p.operation.topology.identity,p.operation.topology.epoch,p.operation.topology.logical_edge_order,{1}};
        check(static_cast<bool>(routes.forward.native.run(values,std::span<const double>(weights).first(p.operation.topology.edge_count),routes.forward.input,physical_input,routes.forward.output,physical_output)));
        success(pk::convert_rows<double>(physical_output,output,routes.forward.destination_order,1,true));
    }
    void vjp(const pk::problem& p,const st::relation_routes& routes) {
        success(pk::convert_rows<double>(cotangent,physical_cotangent,routes.forward.destination_order,1,false));
        pk::nn::value_identity values{routes.input_vjp.descriptor().topology.identity,p.operation.topology.epoch,p.operation.topology.logical_edge_order,{1}};
        check(static_cast<bool>(routes.input_vjp.run(values,std::span<const double>(weights).first(p.operation.topology.edge_count),routes.response_input,physical_cotangent,routes.response_output,physical_gradient)));
        success(pk::convert_rows<double>(physical_gradient,gradient,routes.forward.source_order,1,true));
    }
    void direct_vector(const fixture& f) {
        std::array<double,6> gathered{},products{};
        for(std::size_t i=0;i<6;++i)gathered[i]=input[f.sources[i]];
        check(pk::nn::local_forward(pk::nn::local_operation::multiply,std::span<const double>(weights),std::span<const double>(gathered),std::span<double>(products))==pk::nn::local_status::success);
        output.fill(0);for(std::size_t i=0;i<6;++i)output[f.destinations[i]]+=products[i];
    }
};
