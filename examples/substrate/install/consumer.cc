#include <Cellerator/compute/operation/host_contracts.hh>
#include <Cellerator/compute/operation/native_numeric/host_relation.hh>
#include <Cellerator/compute/operation/indexed_mechanism/incidence.hh>
#include <Cellerator/geometry/evaluator.hh>
#include <ce_moon/reference.hpp>
#include <ce_moon/effects.hpp>
#include <ce_moon/mechanisms.hpp>
#include <learning.hpp>
#include <array>
#include <cmath>
#include <stdexcept>
#include <source_location>
#include <iostream>
namespace nn=cellerator::compute::native_numeric;
namespace ix=cellerator::compute::operation::indexed;
namespace pg=cellerator::execution::program;
void require(bool value, std::source_location location=std::source_location::current()) {
    if(!value) throw std::runtime_error("installed native consumer failed at line "+std::to_string(location.line()));
}
#ifdef CE_REQUIRE_INTEGRATED
void integrated();
#endif
int main() {
    std::array<double,2> x{2,3},p{4,5},out{};
    require(nn::local_forward(nn::local_operation::multiply,std::span<const double>(x),
      std::span<const double>(p),std::span<double>(out))==nn::local_status::success);
    require(out==std::array<double,2>{8,15});
    namespace nf=cellerator::compute::operation::nf1;
    using type=cellerator::execution::numeric_type;
    nf::operand_signature operand{{1,1},{},2,type::f64};
    nf::output_signature output{};output.operand={{2,1},{},2,type::f64};output.assembly_owner={3,1};
    nf::operation_contract contract{};contract.definition={4,1};
    contract.arguments={&operand,1};contract.outputs={&output,1};
    contract.numeric={type::f64,type::f64,type::f64,type::f64,type::f64,type::f64};
    require(nf::validate_operation(contract)==nf::status::success);
    nn::host_relation relation;
    require(!relation.prepare({},{})); // Actual owner rejects absent native identities.
    ix::argument_incidence incidence;
    require(incidence.prepare({},{},{})==ix::incidence_status::success);
    pg::prepared_program_v2 empty{};
    require(pg::execute_prepared_program_v2(empty,nullptr,0,nullptr)==pg::program_status::success);
    require(static_cast<bool>(cellpack::validate_packing_plan_view({})));
    require(ce_moon::compose(ce_moon::Dfa32::identity(),ce_moon::Dfa32::identity()).to[9]==9);
    ce_moon::mechanisms::Matrix matrix(1,1,{2});
    require(ce_moon::mechanisms::solve(matrix,std::vector<double>{6})[0]==3);
    std::array<double,1> row{1};std::array<double,2> weights{0,0};
    require(ce_moon::learning::predict_logistic(row.data(),1,weights.data())==.5);
#ifdef CE_REQUIRE_INTEGRATED
    integrated();
#endif
    std::cout<<"installed actual CE owners passed\n";
}
