#ifdef NDEBUG
#undef NDEBUG
#endif

#include <Cellerator/nn/checkpoint.hh>

#include <ATen/core/grad_mode.h>
#include <torch/optim/adam.h>
#include <torch/serialize/input-archive.h>
#include <torch/serialize/output-archive.h>

#include <cassert>
#include <chrono>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace cellerator::nn;

namespace {

struct OwnerFixture {
    std::unique_ptr<Parameters> parameters;
    ParameterId weight;
    ParameterId tied_weight;
    ParameterId bias;
};

OwnerFixture make_owner(at::ScalarType dtype = at::kFloat,
                       bool tie_weight = true,
                       std::vector<std::int64_t> weight_shape = {2, 2}) {
    OwnerFixture fixture;
    fixture.parameters = std::make_unique<Parameters>(at::Device(at::kCPU), dtype);
    auto weights = fixture.parameters->group();
    fixture.weight = weights.zeros(weight_shape);
    if (tie_weight) {
        auto alias_group = fixture.parameters->group();
        fixture.tied_weight = alias_group.tie(fixture.weight);
    } else {
        auto alias_group = fixture.parameters->group();
        fixture.tied_weight = alias_group.zeros(weight_shape);
    }
    auto bias = fixture.parameters->group();
    fixture.bias = bias.zeros({2});
    return fixture;
}

BiologicalManifest manifest() {
    BiologicalManifest result;
    result.declarations.push_back({
        "",
        "feature_binding",
        R"({"domain":"gene","identities":[{"type":"str","value":"gene-a"}],"canonical_indices":[0],"model_indices":[3],"aliases":[],"metadata":{"reference":"example-v1"},"unknown_policy":"error","unknown_model_index":null,"model_capacity":4})",
    });
    result.relations.push_back({
        "regulation",
        R"({"kind":"cellerator.prepared_relation.v1","source_order":[{"type":"str","value":"gene-a"}],"target_order":[{"type":"str","value":"program-a"}],"source_offsets":[0,1],"target_offsets":[0,1],"source_extent":1,"target_extent":1,"row_ptr":[0,1],"col_indices":[0],"source_axis":null,"target_axis":null,"structural_generation":1})",
    });
    BiologicalViewDeclaration view;
    view.module_path = "readouts";
    view.name = "atac";
    view.output_binding_json =
        R"({"domain":"peak","identities":[{"type":"str","value":"peak-a"}],"canonical_indices":null,"model_indices":[12],"aliases":[],"metadata":{"reference":"hg38"},"unknown_policy":"error","unknown_model_index":null,"model_capacity":16})";
    view.queryable = true;
    view.source_kind = BiologicalViewSourceKind::module_selector;
    view.source_id = "atac-view-v1";
    result.views.push_back(std::move(view));
    return result;
}

void set_values(OwnerFixture& owner, float weight, float bias) {
    at::NoGradGuard no_grad;
    owner.parameters->get(owner.weight).fill_(weight);
    owner.parameters->get(owner.bias).fill_(bias);
}

std::vector<at::Tensor> snapshot_values(const OwnerFixture& owner) {
    std::vector<at::Tensor> values;
    for (const auto& parameter : owner.parameters->torch_parameters())
        values.push_back(parameter.detach().clone());
    return values;
}

void assert_values(const OwnerFixture& owner, const std::vector<at::Tensor>& expected) {
    const auto parameters = owner.parameters->torch_parameters();
    assert(parameters.size() == expected.size());
    for (std::size_t i = 0; i < parameters.size(); ++i)
        assert(at::equal(parameters[i], expected[i]));
}

template<class F>
void rejects(F&& function, const char* message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception&) {
        rejected = true;
    }
    if (!rejected) throw std::runtime_error(message);
}

std::filesystem::path temporary_directory() {
    auto path = std::filesystem::temp_directory_path();
    path /= "cellerator-nn-checkpoint-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    std::filesystem::create_directories(path);
    return path;
}

void rewrite_slot_with_wrong_shape(const std::filesystem::path& source,
                                   const std::filesystem::path& destination,
                                   bool wrong_dtype = false) {
    torch::serialize::InputArchive input;
    input.load_from(source.string());

    c10::IValue format;
    c10::IValue version;
    c10::IValue biological_manifest;
    c10::IValue layout;
    c10::IValue optimizer_layout;
    input.read("format", format);
    input.read("version", version);
    input.read("biological_manifest", biological_manifest);
    input.read("parameter_layout", layout);
    input.read("optimizer_layout", optimizer_layout);

    torch::serialize::InputArchive input_parameters;
    input.read("parameters", input_parameters);
    torch::serialize::OutputArchive output_parameters;
    for (const auto& key : input_parameters.keys()) {
        at::Tensor value;
        input_parameters.read(key, value, false);
        if (key == "slot_0") {
            value = wrong_dtype
                ? at::zeros({2, 2}, at::TensorOptions().dtype(at::kDouble))
                : at::zeros({7}, at::TensorOptions().dtype(at::kFloat));
        }
        output_parameters.write(key, value, false);
    }

    torch::serialize::OutputArchive output;
    output.write("format", format);
    output.write("version", version);
    output.write("biological_manifest", biological_manifest);
    output.write("parameter_layout", layout);
    output.write("optimizer_layout", optimizer_layout);
    output.write("parameters", output_parameters);
    output.save_to(destination.string());
}

void cpu_roundtrip_and_compatible_optimizer_resume(const std::filesystem::path& directory) {
    auto source = make_owner();
    set_values(source, 0.25f, -0.5f);
    torch::optim::Adam source_optimizer(source.parameters->torch_parameters(),
                                        torch::optim::AdamOptions(0.1));
    auto source_loss = source.parameters->get(source.weight).square().sum()
                       + source.parameters->get(source.bias).square().sum();
    source_loss.backward();
    source_optimizer.step();
    source_optimizer.zero_grad();

    const auto checkpoint = directory / "model.pt";
    save_checkpoint(checkpoint, *source.parameters, manifest(), &source_optimizer);
    const auto saved_values = snapshot_values(source);
    assert(source.parameters->get(source.weight).is_same(
        source.parameters->get(source.tied_weight)));
    assert(source_optimizer.state().size() == 2);

    auto target = make_owner();
    set_values(target, 7.0f, 9.0f);
    torch::optim::Adam target_optimizer(target.parameters->torch_parameters(),
                                        torch::optim::AdamOptions(0.7));
    load_checkpoint(checkpoint, *target.parameters, manifest(), &target_optimizer);
    assert_values(target, saved_values);
    assert(target.parameters->get(target.weight).is_same(
        target.parameters->get(target.tied_weight)));
    assert(target_optimizer.state().size() == 2);
    assert(static_cast<torch::optim::AdamOptions&>(
        target_optimizer.param_groups()[0].options()).lr() == 0.1);

    source_loss = source.parameters->get(source.weight).square().sum()
                  + source.parameters->get(source.bias).square().sum();
    source_loss.backward();
    source_optimizer.step();
    source_optimizer.zero_grad();
    auto target_loss = target.parameters->get(target.weight).square().sum()
                       + target.parameters->get(target.bias).square().sum();
    target_loss.backward();
    target_optimizer.step();
    target_optimizer.zero_grad();
    assert_values(target, snapshot_values(source));
}

void incompatibilities_reject_before_parameter_mutation(const std::filesystem::path& directory) {
    auto source = make_owner();
    set_values(source, 1.5f, -2.0f);
    const auto checkpoint = directory / "semantic.pt";
    save_checkpoint(checkpoint, *source.parameters, manifest());

    auto target = make_owner();
    set_values(target, 11.0f, 12.0f);
    const auto before = snapshot_values(target);
    auto changed_manifest = manifest();
    changed_manifest.declarations[0].descriptor_json =
        R"({"domain":"gene","identities":[{"type":"str","value":"gene-b"}],"canonical_indices":[0],"model_indices":[3],"aliases":[],"metadata":{"reference":"example-v1"},"unknown_policy":"error","unknown_model_index":null,"model_capacity":4})";
    rejects([&] {
        load_checkpoint(checkpoint, *target.parameters, changed_manifest);
    }, "biological declaration mismatch must reject");
    assert_values(target, before);

    auto changed_relation = manifest();
    changed_relation.relations[0].descriptor_json =
        R"({"kind":"cellerator.prepared_relation.v1","source_order":[{"type":"str","value":"gene-b"}],"target_order":[{"type":"str","value":"program-a"}],"source_offsets":[0,1],"target_offsets":[0,1],"source_extent":1,"target_extent":1,"row_ptr":[0,1],"col_indices":[0],"source_axis":null,"target_axis":null,"structural_generation":1})";
    rejects([&] {
        load_checkpoint(checkpoint, *target.parameters, changed_relation);
    }, "biological relation endpoints mismatch must reject");
    assert_values(target, before);

    auto changed_view = manifest();
    changed_view.views[0].source_id = "another-atac-selector-v1";
    rejects([&] {
        load_checkpoint(checkpoint, *target.parameters, changed_view);
    }, "biological view selector mismatch must reject");
    assert_values(target, before);

    auto wrong_aliases = make_owner(at::kFloat, false);
    set_values(wrong_aliases, 13.0f, 14.0f);
    const auto alias_before = snapshot_values(wrong_aliases);
    rejects([&] {
        load_checkpoint(checkpoint, *wrong_aliases.parameters, manifest());
    }, "alias layout mismatch must reject");
    assert_values(wrong_aliases, alias_before);

    auto wrong_shape = make_owner(at::kFloat, true, {4});
    set_values(wrong_shape, 15.0f, 16.0f);
    const auto shape_before = snapshot_values(wrong_shape);
    rejects([&] {
        load_checkpoint(checkpoint, *wrong_shape.parameters, manifest());
    }, "slot shape mismatch must reject");
    assert_values(wrong_shape, shape_before);

    auto wrong_dtype = make_owner(at::kDouble);
    set_values(wrong_dtype, 17.0f, 18.0f);
    const auto dtype_before = snapshot_values(wrong_dtype);
    rejects([&] {
        load_checkpoint(checkpoint, *wrong_dtype.parameters, manifest());
    }, "slot dtype mismatch must reject");
    assert_values(wrong_dtype, dtype_before);

    torch::optim::Adam source_optimizer(source.parameters->torch_parameters(),
                                        torch::optim::AdamOptions(0.1));
    const auto optimizer_checkpoint = directory / "optimizer-groups.pt";
    save_checkpoint(optimizer_checkpoint, *source.parameters, manifest(), &source_optimizer);
    auto wrong_optimizer_groups = make_owner();
    set_values(wrong_optimizer_groups, 19.0f, 20.0f);
    std::vector<torch::optim::OptimizerParamGroup> parameter_groups;
    parameter_groups.emplace_back(std::vector<at::Tensor>{
        wrong_optimizer_groups.parameters->get(wrong_optimizer_groups.weight)});
    parameter_groups.emplace_back(std::vector<at::Tensor>{
        wrong_optimizer_groups.parameters->get(wrong_optimizer_groups.bias)});
    torch::optim::Adam target_optimizer(std::move(parameter_groups),
                                        torch::optim::AdamOptions(0.9));
    const auto optimizer_before = snapshot_values(wrong_optimizer_groups);
    rejects([&] {
        load_checkpoint(optimizer_checkpoint, *wrong_optimizer_groups.parameters,
                       manifest(), &target_optimizer);
    }, "optimizer parameter groups must match numeric checkpoint slots");
    assert_values(wrong_optimizer_groups, optimizer_before);
}

void serialized_tensor_shape_is_staged_before_copy(const std::filesystem::path& directory) {
    auto source = make_owner();
    set_values(source, 1.0f, 2.0f);
    const auto checkpoint = directory / "valid.pt";
    const auto malformed = directory / "wrong-shape.pt";
    const auto bad_dtype = directory / "wrong-dtype.pt";
    save_checkpoint(checkpoint, *source.parameters, manifest());
    rewrite_slot_with_wrong_shape(checkpoint, malformed);
    rewrite_slot_with_wrong_shape(checkpoint, bad_dtype, true);

    auto target = make_owner();
    set_values(target, -7.0f, -8.0f);
    const auto before = snapshot_values(target);
    rejects([&] {
        load_checkpoint(malformed, *target.parameters, manifest());
    }, "malformed slot tensor shape must reject");
    assert_values(target, before);
    rejects([&] {
        load_checkpoint(bad_dtype, *target.parameters, manifest());
    }, "malformed slot tensor dtype must reject");
    assert_values(target, before);
}

void selected_group_roundtrip_preserves_other_groups_and_resumes_adam(
    const std::filesystem::path& directory) {
    Parameters source(at::Device(at::kCPU), at::kFloat);
    auto source_selected = source.group();
    const auto source_weight = source_selected.zeros({2, 2});
    auto source_other = source.group();
    const auto source_other_slot = source_other.zeros({1});
    const auto source_bias = source_selected.zeros({2});
    const auto source_tied = source_selected.tie(source_weight);
    assert(source_selected.parameter_ids().size() == 3);
    assert(source_selected.parameter_ids()[1].value == 2);
    {
        at::NoGradGuard no_grad;
        source_selected.get(source_weight).fill_(0.25f);
        source_selected.get(source_bias).fill_(-0.5f);
        source_other.get(source_other_slot).fill_(21.0f);
    }
    torch::optim::Adam source_optimizer(source_selected.torch_parameters(),
                                        torch::optim::AdamOptions(0.1));
    auto source_loss = source_selected.get(source_weight).square().sum()
                       + source_selected.get(source_bias).square().sum();
    source_loss.backward();
    source_optimizer.step();
    source_optimizer.zero_grad();

    const auto checkpoint = directory / "selected-group.pt";
    save_checkpoint(checkpoint, source_selected, manifest(), &source_optimizer);
    const auto saved_weight = source_selected.get(source_weight).detach().clone();
    const auto saved_bias = source_selected.get(source_bias).detach().clone();

    Parameters target(at::Device(at::kCPU), at::kFloat);
    auto target_selected = target.group();
    const auto target_weight = target_selected.zeros({2, 2});
    auto target_other = target.group();
    const auto target_other_slot = target_other.zeros({1});
    const auto target_bias = target_selected.zeros({2});
    const auto target_tied = target_selected.tie(target_weight);
    {
        at::NoGradGuard no_grad;
        target_selected.get(target_weight).fill_(7.0f);
        target_selected.get(target_bias).fill_(8.0f);
        target_other.get(target_other_slot).fill_(99.0f);
    }
    const auto target_weight_handle = target_selected.get(target_weight);
    const auto target_bias_handle = target_selected.get(target_bias);
    const auto target_other_handle = target_other.get(target_other_slot);
    const auto selected_before = std::vector<at::Tensor>{
        target_selected.get(target_weight).detach().clone(),
        target_selected.get(target_bias).detach().clone(),
    };
    const auto other_before = target_other.get(target_other_slot).detach().clone();
    torch::optim::Adam wrong_target_optimizer(target.torch_parameters(),
                                              torch::optim::AdamOptions(0.9));
    rejects([&] {
        load_checkpoint(checkpoint, target_selected, manifest(),
                        &wrong_target_optimizer);
    }, "selected group load optimizer must not contain outside leaves");
    assert(at::equal(target_selected.get(target_weight), selected_before[0]));
    assert(at::equal(target_selected.get(target_bias), selected_before[1]));
    assert(at::equal(target_other.get(target_other_slot), other_before));

    torch::optim::Adam target_optimizer(target_selected.torch_parameters(),
                                        torch::optim::AdamOptions(0.9));
    load_checkpoint(checkpoint, target_selected, manifest(), &target_optimizer);
    assert(at::equal(target_selected.get(target_weight), saved_weight));
    assert(at::equal(target_selected.get(target_bias), saved_bias));
    assert(target_selected.get(target_weight).is_same(target_selected.get(target_tied)));
    assert(target_selected.get(target_weight).is_same(target_weight_handle));
    assert(target_selected.get(target_bias).is_same(target_bias_handle));
    assert(target_other.get(target_other_slot).is_same(target_other_handle));
    assert(at::equal(target_other.get(target_other_slot), other_before));
    assert(target_optimizer.state().size() == 2);
    assert(static_cast<torch::optim::AdamOptions&>(
        target_optimizer.param_groups()[0].options()).lr() == 0.1);

    source_loss = source_selected.get(source_weight).square().sum()
                  + source_selected.get(source_bias).square().sum();
    source_loss.backward();
    source_optimizer.step();
    source_optimizer.zero_grad();
    auto target_loss = target_selected.get(target_weight).square().sum()
                       + target_selected.get(target_bias).square().sum();
    target_loss.backward();
    target_optimizer.step();
    target_optimizer.zero_grad();
    assert(at::equal(target_selected.get(target_weight),
                     source_selected.get(source_weight)));
    assert(at::equal(target_selected.get(target_bias),
                     source_selected.get(source_bias)));
    assert(at::equal(target_other.get(target_other_slot), other_before));

    torch::optim::Adam wrong_optimizer(source.torch_parameters(),
                                       torch::optim::AdamOptions(0.1));
    rejects([&] {
        save_checkpoint(directory / "wrong-selected-optimizer.pt",
                        source_selected, manifest(), &wrong_optimizer);
    }, "selected group optimizer must not contain leaves outside the selection");
}

} // namespace

int main() {
    const auto directory = temporary_directory();
    try {
        cpu_roundtrip_and_compatible_optimizer_resume(directory);
        incompatibilities_reject_before_parameter_mutation(directory);
        serialized_tensor_shape_is_staged_before_copy(directory);
        selected_group_roundtrip_preserves_other_groups_and_resumes_adam(directory);
    } catch (...) {
        std::filesystem::remove_all(directory);
        throw;
    }
    std::filesystem::remove_all(directory);
}
