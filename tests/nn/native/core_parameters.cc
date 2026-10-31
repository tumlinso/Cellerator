#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/nn/parameters.hh>

#include <ATen/ATen.h>
#include <torch/optim.h>

#include <cassert>
#include <stdexcept>
#include <vector>

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
    const auto shared_ids = shared.parameter_ids();
    assert(shared_ids.size() == 2 && shared_ids[0] == weight_id
           && shared_ids[1] == tied_id);
    const auto shared_tensors = shared.torch_parameters();
    assert(shared_tensors.size() == 1);
    assert(shared_tensors[0].unsafeGetTensorImpl()
           == owner.get(weight_id).unsafeGetTensorImpl());

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
    const auto frozen_tensors = frozen.torch_parameters();
    assert(frozen_tensors.size() == 1);
    assert(frozen_tensors[0].unsafeGetTensorImpl()
           == owner.get(frozen_id).unsafeGetTensorImpl());
    assert(owner.group_layout().size() == 2);
    assert(owner.layout()[2].group_id == frozen.id());
    assert(!owner.layout()[2].trainable);

    frozen.set_trainable(true);
    assert(owner.get(frozen_id).requires_grad());

    frozen.set_trainable(false);
    auto alias_group = owner.group();
    const auto frozen_alias = alias_group.tie(frozen_id);
    assert(!owner.get(frozen_alias).requires_grad());
    assert(owner.get(frozen_id).unsafeGetTensorImpl()
           == owner.get(frozen_alias).unsafeGetTensorImpl());
    alias_group.set_trainable(true);
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

    Parameters adoption(at::Device(at::kCPU), at::kFloat);
    auto original = adoption.group();
    auto external = at::ones({2, 2}); // An external frozen leaf stays frozen on add.
    const auto external_id = original.add(external);
    assert(adoption.get(external_id).unsafeGetTensorImpl()
           == external.unsafeGetTensorImpl());
    assert(!adoption.get(external_id).requires_grad());

    auto alias_owner = adoption.group();
    const auto duplicate_id = alias_owner.add(external);
    assert(adoption.get(duplicate_id).unsafeGetTensorImpl()
           == external.unsafeGetTensorImpl());
    assert(adoption.layout()[duplicate_id.value].tied_to == external_id);
    assert(adoption.torch_parameters().size() == 1);
    alias_owner.set_trainable(true);
    assert(external.requires_grad());

    auto wrong_dtype = at::ones({2, 2}, at::TensorOptions().dtype(at::kDouble));
    rejected = false;
    try { (void)original.add(wrong_dtype); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected && adoption.size() == 2);

    auto nonleaf_source = at::ones({2});
    nonleaf_source.requires_grad_(true);
    auto nonleaf = nonleaf_source * 2.0;
    assert(!nonleaf.is_leaf());
    rejected = false;
    try { (void)original.add(nonleaf); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected && adoption.size() == 2);

    Parameters selection(at::Device(at::kCPU), at::kFloat);
    auto selected_group = selection.group();
    auto unrelated_group = selection.group();
    const auto selected_id = selected_group.zeros({2});
    const auto unrelated_id = unrelated_group.zeros({2});
    const auto selected_values = selection.get(selected_id);
    const auto unrelated_values = selection.get(unrelated_id);
    const auto selected_before = selected_values.clone().detach();
    const auto unrelated_before = unrelated_values.clone().detach();
    (selected_values.sum() + unrelated_values.sum()).backward();

    torch::optim::SGD selected_optimizer(
        selected_group.torch_parameters(), torch::optim::SGDOptions(0.1));
    selected_optimizer.step();
    assert(!at::equal(selected_before, selected_values));
    assert(at::equal(unrelated_before, unrelated_values));
    return 0;
}
