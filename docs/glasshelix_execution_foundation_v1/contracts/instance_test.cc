#include <Cellerator/compute/operation/native_foundation_contract.hh>
#include <cassert>
namespace nf = cellerator::compute::operation::nf1;
int main() {
    nf::prepared_identity prepared{{1, 0}, {2, 0}, {1}, {3, 0}};
    nf::instance_binding a{prepared, {{10, 0}, {2}}, {{20, 0}, {5}}};
    auto b = a; b.state.instance.low = 11; b.parameters = {{21, 0}, {8}};
    assert(nf::validate_instance(prepared, a) == nf::status::success);
    assert(nf::validate_instance(prepared, b) == nf::status::success);
    assert(!nf::explicitly_tied(a, b));
    b.parameters = a.parameters; // Even equal generation identity does not infer tying.
    assert(!nf::explicitly_tied(a, b));
    a.parameter_tie_group = b.parameter_tie_group = {99, 0};
    assert(nf::explicitly_tied(a, b));
    b.prepared.epoch.value++;
    assert(nf::validate_instance(prepared, b) == nf::status::invalid_binding);
    b = a; b.state.generation.value = 0;
    assert(nf::validate_instance(prepared, b) == nf::status::stale_generation);
}
