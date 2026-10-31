#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/nn/parameters.hh>

#include <ATen/ATen.h>

#include <cassert>
#include <stdexcept>

using namespace cellerator::nn;

int main() {
    Parameters owner(at::Device(at::kCPU), at::kFloat);
    auto shared = owner.group();
    const auto weight_id = shared.zeros({2, 3});
    const auto tied_id = shared.tie(weight_id);

    assert(weight_id.value == 0 && tied_id.value == 1);
    assert(owner.size() == 2);
    assert(owner.get(weight_id).sizes().vec() == (std::vector<std::int64_t>{2, 3}));
    assert(owner.get(weight_id).unsafeGetTensorImpl()
           == owner.get(tied_id).unsafeGetTensorImpl());
    assert(owner.torch_parameters().size() == 1);

    const auto slots = owner.layout();
    assert(slots.size() == 2);
    assert(slots[0].id == weight_id && slots[0].group_id == shared.id());
    assert(slots[0].trainable && !slots[0].tied_to);
    assert(slots[1].id == tied_id && slots[1].tied_to == weight_id);
    assert(slots[1].group_id == shared.id());

    auto frozen = owner.group(at::TensorOptions().dtype(at::kDouble));
    frozen.set_trainable(false);
    const auto frozen_id = frozen.normal({4}, 0.25);
    auto frozen_tensor = owner.get(frozen_id);
    assert(frozen_tensor.device().is_cpu());
    assert(frozen_tensor.scalar_type() == at::kDouble);
    assert(!frozen_tensor.requires_grad());
    assert(owner.torch_parameters().size() == 2); // Includes frozen leaves.
    assert(owner.group_layout().size() == 2);
    assert(owner.layout()[2].group_id == frozen.id());
    assert(!owner.layout()[2].trainable);

    frozen.set_trainable(true);
    assert(owner.get(frozen_id).requires_grad());

    bool rejected = false;
    try { frozen.set_options(at::TensorOptions().dtype(at::kFloat)); }
    catch (const std::logic_error&) { rejected = true; }
    assert(rejected);

    rejected = false;
    try { (void)owner.get(ParameterId{999}); }
    catch (const std::out_of_range&) { rejected = true; }
    assert(rejected);

    rejected = false;
    try { Parameters invalid(at::Device(at::kCPU), at::kLong); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    return 0;
}
