import cellerator.nn;

static_assert(sizeof(at::Tensor) > 0);
static_assert(sizeof(at::Device) > 0);

struct CellProbe : cellerator::CellAccess<int> {
protected:
    int read(cellerator::ViewTag<int>) const & override { return 17; }
};

bool module_only_surface_works() {
    CellProbe probe;
    const cellerator::Cell& cell = probe;
    if (!cell.supports<int>() || cell.view<int>() != 17 || cell.supports<double>()) {
        return false;
    }
    int tokens[] = {3, 3};
    const cellerator::memory::const_array_view<int> buffer{tokens, 2, {}};
    cellerator::cells::EventCell<int, int> event(buffer, 0, 2, 9);
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
