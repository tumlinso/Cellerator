#include <Cellerator/nn.hh>

#include "cell_bridge_contract.hh"

#include <cstddef>
#include <typeinfo>

ProbeCell header_make_cell(ProbeModel& model, int value) {
    return cellerator::make_cell(model, ProbeInstance{value});
}

int header_cell_forward(ProbeCell& cell, int increment) {
    return cell.forward(increment);
}

bool header_cell_view_matches(const ProbeCell& cell, int expected) {
    return cell.view<ProbeView>().value == expected;
}

bool header_runtime_view_matches(const cellerator::CellViews& views, int expected) {
    if (!views.supports<ProbeView>() || views.supports<double>() ||
        views.view<ProbeView>().value != expected) {
        return false;
    }
    try {
        (void)views.view<double>();
    } catch (const std::bad_cast&) {
        return true;
    }
    return false;
}

// This ordinary-header translation unit shares real NN and ATen types with
// the importing and, optionally, NVCC-compiled translation units.
std::size_t header_owner_slots(const cellerator::nn::Parameters& owner) {
    return owner.size();
}

bool header_tensor_defined(const at::Tensor& value) {
    return value.defined();
}
