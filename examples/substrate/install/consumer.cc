#include <Cellerator/execution/preparation/host_contracts.hh>
#include <Cellerator/ops/relation/numeric/host_relation.hh>
#include <Cellerator/geometry/packing/evaluator.hh>
#include <Cellerator/ops/mechanism/indexed/incidence.hh>

namespace nn = cellerator::ops::relation::numeric;
namespace ix = cellerator::ops::mechanism::indexed;
namespace pg = cellerator::execution::program;
namespace nf = cellerator::execution::operation::nf1;
using type = cellerator::execution::numeric_type;

#ifdef CE_REQUIRE_INTEGRATED
void integrated();
#endif

int main() {
    const double x[]{2, 3}, p[]{4, 5};
    double out[2]{};
    if (nn::local_forward(nn::local_operation::multiply, x, p, out)
        != nn::local_status::success || out[0] != 8 || out[1] != 15) return 1;

    nf::operand_signature operand{{1, 1}, {}, 2, type::f64};
    nf::output_signature output{};
    output.operand = {{2, 1}, {}, 2, type::f64};
    output.assembly_owner = {3, 1};
    nf::operation_contract contract{};
    contract.definition = {4, 1};
    contract.arguments = {&operand, 1};
    contract.outputs = {&output, 1};
    contract.numeric = {type::f64, type::f64, type::f64, type::f64, type::f64, type::f64};
    if (nf::validate_operation(contract) != nf::status::success) return 2;

    nn::host_relation relation;
    if (relation.prepare({}, {})) return 3;
    ix::argument_incidence incidence;
    if (incidence.prepare({}, {}, {}) != ix::incidence_status::success) return 4;
    pg::prepared_program_v2 empty{};
    if (pg::execute_prepared_program_v2(empty, nullptr, 0, nullptr)
        != pg::program_status::success) return 5;
    if (!cellpack::validate_packing_plan_view({})) return 6;
#ifdef CE_REQUIRE_INTEGRATED
    integrated();
#endif
    return 0;
}
