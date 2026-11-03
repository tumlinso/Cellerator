import cellerator.nn;

#include "cell_bridge_contract.hh"

static_assert(sizeof(at::Tensor) > 0);
static_assert(sizeof(at::Device) > 0);

struct EventModel {};

bool module_only_surface_works() {
    ProbeModel model;
    auto probe = cellerator::make_cell(model, ProbeInstance{17});
    auto bridge = cellerator::make_cell_views<
        cellerator::CellAccess<ProbeView>>(probe);
    cellerator::CellViews& views = bridge;
    if (!views.supports<ProbeView>() || views.view<ProbeView>().value != 17 ||
        views.supports<double>() || probe.forward(1) != 18) {
        return false;
    }
    // Pass canonical typed templates and the optional RTTI interface across
    // an import/header translation-unit boundary in both directions.
    if (header_cell_forward(probe, 2) != 20 ||
        !header_cell_view_matches(probe, 20) ||
        !header_runtime_view_matches(views, 20)) {
        return false;
    }
    auto from_header = header_make_cell(model, 4);
    if (from_header.forward(3) != 7) {
        return false;
    }

    int tokens[] = {3, 3};
    const cellerator::memory::const_array_view<int> buffer{tokens, 2, {}};
    EventModel event_model;
    auto event = cellerator::make_cell(event_model,
        cellerator::cells::EventPrefix<int, int>(buffer, 0, 2, 9));
    if (event.view<cellerator::cells::EventView<int>>().tokens.count != 2 ||
        event.view<cellerator::CellIdentityView<int>>().value != 9) {
        return false;
    }

    cellerator::nn::Parameters parameters(
        at::Device(at::DeviceType::CPU), at::ScalarType::Float);
    const at::Tensor undefined_tensor;
    if (undefined_tensor.defined()) {
        return false;
    }
    return parameters.size() == 0;
}
