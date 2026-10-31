#include <Cellerator/nn/checkpoint.hh>

#include <nlohmann/json.hpp>

#include <ATen/core/grad_mode.h>
#include <c10/core/TensorImpl.h>
#include <torch/optim/optimizer.h>
#include <torch/serialize/input-archive.h>
#include <torch/serialize/output-archive.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace cellerator::nn {
namespace {

using Json = nlohmann::json;
using TensorImpl = c10::TensorImpl;

struct ParameterSnapshot {
    std::vector<ParameterSlot> slots;
    std::vector<ParameterGroupLayout> groups;
    std::map<std::uint64_t, ParameterSlot> slots_by_id;
    std::vector<ParameterId> roots;
    std::unordered_map<const TensorImpl*, ParameterId> root_by_tensor;
};

std::string dtype_name(at::ScalarType dtype) {
    return c10::toString(dtype);
}

ParameterSnapshot snapshot_parameters(const Parameters& parameters) {
    ParameterSnapshot snapshot;
    snapshot.slots = parameters.layout();
    snapshot.groups = parameters.group_layout();

    std::map<std::uint64_t, std::size_t> group_index;
    for (std::size_t i = 0; i < snapshot.groups.size(); ++i) {
        const auto& group = snapshot.groups[i];
        if (!group_index.emplace(group.id, i).second)
            throw std::logic_error("parameter owner has duplicate numeric group IDs");
        if (!group.options.dtype_opt())
            throw std::logic_error("parameter group layout must declare a dtype");
        (void)group.options.dtype_opt()->toScalarType();
    }

    std::unordered_map<const TensorImpl*, std::uint64_t> first_slot_by_tensor;
    for (const auto& slot : snapshot.slots) {
        if (!slot.id.valid())
            throw std::logic_error("parameter owner has an invalid numeric slot ID");
        if (!snapshot.slots_by_id.emplace(slot.id.value, slot).second)
            throw std::logic_error("parameter owner has duplicate numeric slot IDs");
        if (!group_index.count(slot.group_id))
            throw std::logic_error("parameter slot refers to a missing group");

        auto tensor = parameters.get(slot.id);
        if (!tensor.defined() || tensor.sizes().vec() != slot.shape ||
            tensor.scalar_type() != slot.dtype || tensor.device() != slot.device ||
            tensor.requires_grad() != slot.trainable)
            throw std::logic_error("parameter slot metadata disagrees with its leaf tensor");

        std::uint64_t canonical_id = slot.id.value;
        if (slot.tied_to) {
            auto tied = snapshot.slots_by_id.find(slot.tied_to->value);
            if (tied == snapshot.slots_by_id.end())
                throw std::logic_error("tied parameter refers to a missing slot");
            if (tied->second.tied_to)
                throw std::logic_error("tied parameter must refer directly to its canonical slot");
            canonical_id = tied->first;
            auto canonical_tensor = parameters.get(tied->second.id);
            if (!tensor.is_same(canonical_tensor))
                throw std::logic_error("tied parameter slots do not share the same leaf tensor");
            if (slot.shape != tied->second.shape || slot.dtype != tied->second.dtype)
                throw std::logic_error("tied parameter slot shape or dtype differs from its canonical slot");
        } else {
            const auto* impl = tensor.unsafeGetTensorImpl();
            if (!first_slot_by_tensor.emplace(impl, slot.id.value).second)
                throw std::logic_error("parameter owner has an undeclared tied leaf tensor");
            snapshot.roots.push_back(slot.id);
            snapshot.root_by_tensor.emplace(impl, slot.id);
        }
        (void)canonical_id;
    }

    std::map<std::uint64_t, std::size_t> membership_count;
    for (const auto& group : snapshot.groups) {
        for (const auto id : group.slots) {
            auto slot = snapshot.slots_by_id.find(id.value);
            if (slot == snapshot.slots_by_id.end())
                throw std::logic_error("parameter group refers to a missing slot");
            if (slot->second.group_id != group.id)
                throw std::logic_error("parameter slot and group layout disagree");
            if (++membership_count[id.value] != 1)
                throw std::logic_error("parameter slot belongs to multiple group layouts");
        }
    }
    for (const auto& slot : snapshot.slots) {
        if (membership_count[slot.id.value] != 1)
            throw std::logic_error("parameter slot is missing from its group layout");
    }
    return snapshot;
}

Json parameter_layout_json(const ParameterSnapshot& snapshot) {
    Json groups = Json::array();
    for (const auto& group : snapshot.groups) {
        Json slots = Json::array();
        for (const auto id : group.slots) slots.push_back(id.value);
        groups.push_back({
            {"id", group.id},
            {"dtype", dtype_name(group.options.dtype_opt()->toScalarType())},
            {"trainable", group.trainable},
            {"slots", std::move(slots)},
        });
    }
    Json slots = Json::array();
    for (const auto& slot : snapshot.slots) {
        Json shape = Json::array();
        for (const auto dim : slot.shape) shape.push_back(dim);
        slots.push_back({
            {"id", slot.id.value},
            {"group_id", slot.group_id},
            {"shape", std::move(shape)},
            {"dtype", dtype_name(slot.dtype)},
            {"trainable", slot.trainable},
            {"tied_to", slot.tied_to ? Json(slot.tied_to->value) : Json(nullptr)},
        });
    }
    return Json{{"groups", std::move(groups)}, {"slots", std::move(slots)}};
}

std::string parameter_layout_string(const ParameterSnapshot& snapshot) {
    return parameter_layout_json(snapshot).dump(-1, ' ', false);
}

Json optimizer_layout_json(
    const ParameterSnapshot& parameters,
    const torch::optim::Optimizer* optimizer) {
    Json groups = Json::array();
    if (optimizer) {
        for (const auto& group : optimizer->param_groups()) {
            Json slots = Json::array();
            for (const auto& tensor : group.params()) {
                if (!tensor.defined())
                    throw std::invalid_argument("optimizer contains an undefined parameter tensor");
                const auto found = parameters.root_by_tensor.find(tensor.unsafeGetTensorImpl());
                if (found == parameters.root_by_tensor.end())
                    throw std::invalid_argument("optimizer parameter is not owned by nn::Parameters");
                slots.push_back(found->second.value);
            }
            groups.push_back(std::move(slots));
        }
    }
    return Json{{"present", optimizer != nullptr}, {"groups", std::move(groups)}};
}

std::string optimizer_layout_string(
    const ParameterSnapshot& parameters,
    const torch::optim::Optimizer* optimizer) {
    return optimizer_layout_json(parameters, optimizer).dump(-1, ' ', false);
}

std::string slot_archive_key(ParameterId id) {
    return "slot_" + std::to_string(id.value);
}

void write_string(torch::serialize::OutputArchive& archive,
                  const std::string& key,
                  const std::string& value) {
    archive.write(key, c10::IValue(value));
}

std::string read_string(torch::serialize::InputArchive& archive,
                        const std::string& key) {
    c10::IValue value;
    archive.read(key, value);
    if (!value.isString())
        throw std::invalid_argument("checkpoint archive field '" + key + "' must be a string");
    return std::string(value.toStringRef());
}

void require_keys(torch::serialize::InputArchive& archive,
                  const std::set<std::string>& expected,
                  const char* description) {
    const auto keys = archive.keys();
    const std::set<std::string> actual(keys.begin(), keys.end());
    if (actual != expected)
        throw std::invalid_argument(std::string("checkpoint ") + description + " keys do not match the schema");
}

std::filesystem::path temporary_path(const std::filesystem::path& path) {
    static std::atomic<std::uint64_t> sequence{0};
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    auto result = path;
    result += ".tmp." + std::to_string(stamp) + "." +
              std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
    return result;
}

void copy_values(const std::vector<std::pair<at::Tensor, at::Tensor>>& values) {
    at::NoGradGuard no_grad;
    for (const auto& entry : values) entry.first.copy_(entry.second);
}

Parameters selected_group_parameters(const ParameterGroup& selected) {
    const auto ids = selected.parameter_ids();
    at::Device base_device(at::kCPU);
    at::ScalarType base_dtype = at::kFloat;
    if (!ids.empty()) {
        const auto first = selected.get(ids.front());
        base_device = first.device();
        base_dtype = first.scalar_type();
    } else {
        const auto options = selected.options();
        if (options.device_opt()) base_device = *options.device_opt();
        if (options.dtype_opt()) base_dtype = options.dtype_opt()->toScalarType();
    }

    Parameters owner(base_device, base_dtype);
    for (const auto id : ids) {
        const auto tensor = selected.get(id);
        auto slot_group = owner.group(tensor.options());
        // ParameterGroup::add adopts this exact leaf. A repeated TensorImpl
        // becomes a local alias slot, while a leaf outside the selected group
        // remains a root in this local checkpoint view.
        (void)slot_group.add(tensor);
    }
    return owner;
}

} // namespace

void save_checkpoint(
    const std::filesystem::path& path,
    const Parameters& parameters,
    const BiologicalManifest& manifest,
    torch::optim::Optimizer* optimizer) {
    if (manifest.weights)
        throw std::invalid_argument("native checkpoint manifests must use weights=null");
    const auto manifest_json = serialize_biological_manifest(manifest);
    const auto snapshot = snapshot_parameters(parameters);
    const auto layout_json = parameter_layout_string(snapshot);
    const auto optimizer_json = optimizer_layout_string(snapshot, optimizer);

    torch::serialize::OutputArchive archive;
    archive.write("format", c10::IValue(std::string(native_checkpoint_format)));
    archive.write("version", c10::IValue(static_cast<std::int64_t>(native_checkpoint_version)));
    write_string(archive, "biological_manifest", manifest_json);
    write_string(archive, "parameter_layout", layout_json);
    write_string(archive, "optimizer_layout", optimizer_json);

    torch::serialize::OutputArchive slot_archive;
    {
        at::NoGradGuard no_grad;
        for (const auto id : snapshot.roots) {
            auto value = parameters.get(id).detach().to(at::Device(at::kCPU)).contiguous().clone();
            slot_archive.write(slot_archive_key(id), value, false);
        }
    }
    archive.write("parameters", slot_archive);

    if (optimizer) {
        torch::serialize::OutputArchive optimizer_archive;
        optimizer->save(optimizer_archive);
        archive.write("optimizer", optimizer_archive);
    }

    const auto parent = path.has_parent_path() ? path.parent_path() : std::filesystem::path(".");
    std::filesystem::create_directories(parent);
    const auto temporary = temporary_path(path);
    try {
        archive.save_to(temporary.string());
        std::filesystem::rename(temporary, path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

void save_checkpoint(
    const std::filesystem::path& path,
    const ParameterGroup& group,
    const BiologicalManifest& manifest,
    torch::optim::Optimizer* optimizer) {
    auto owner = selected_group_parameters(group);
    save_checkpoint(path, owner, manifest, optimizer);
}

void load_checkpoint(
    const std::filesystem::path& path,
    Parameters& parameters,
    const BiologicalManifest& expected_manifest,
    torch::optim::Optimizer* optimizer) {
    if (expected_manifest.weights)
        throw std::invalid_argument("native checkpoint expected manifests must use weights=null");

    const auto snapshot = snapshot_parameters(parameters);
    const auto expected_layout = parameter_layout_string(snapshot);
    const auto expected_optimizer_layout = optimizer_layout_string(snapshot, optimizer);

    torch::serialize::InputArchive archive;
    archive.load_from(path.string());

    const auto optimizer_json = read_string(archive, "optimizer_layout");
    const auto optimizer_metadata = Json::parse(optimizer_json);
    if (!optimizer_metadata.is_object() || !optimizer_metadata.contains("present") ||
        !optimizer_metadata.at("present").is_boolean() || !optimizer_metadata.contains("groups") ||
        !optimizer_metadata.at("groups").is_array())
        throw std::invalid_argument("checkpoint optimizer layout is malformed");
    const bool saved_optimizer = optimizer_metadata.at("present").get<bool>();
    for (const auto& group : optimizer_metadata.at("groups")) {
        if (!group.is_array())
            throw std::invalid_argument("checkpoint optimizer groups must be arrays of numeric slots");
        for (const auto& id : group) {
            if (!id.is_number_unsigned() && !id.is_number_integer())
                throw std::invalid_argument("checkpoint optimizer slot IDs must be integers");
            if (!id.is_number_unsigned() && id.get<std::int64_t>() < 0)
                throw std::invalid_argument("checkpoint optimizer slot IDs must be nonnegative");
        }
    }
    if (!saved_optimizer && !optimizer_metadata.at("groups").empty())
        throw std::invalid_argument("checkpoint without an optimizer must have empty optimizer groups");

    std::set<std::string> archive_keys{
        "format", "version", "biological_manifest", "parameter_layout",
        "optimizer_layout", "parameters",
    };
    if (saved_optimizer) archive_keys.insert("optimizer");
    require_keys(archive, archive_keys, "archive");

    c10::IValue format;
    archive.read("format", format);
    if (!format.isString() ||
        std::string(format.toStringRef()) != std::string(native_checkpoint_format))
        throw std::invalid_argument("unsupported Cellerator native NN checkpoint format");
    c10::IValue version;
    archive.read("version", version);
    if (!version.isInt() || version.toInt() != native_checkpoint_version)
        throw std::invalid_argument("unsupported Cellerator native NN checkpoint version");

    const auto stored_manifest = read_string(archive, "biological_manifest");
    validate_biological_manifest(stored_manifest, expected_manifest);
    if (parse_biological_manifest(stored_manifest).weights)
        throw std::invalid_argument("native checkpoint manifest must use weights=null");

    const auto saved_layout = read_string(archive, "parameter_layout");
    if (saved_layout != expected_layout)
        throw std::invalid_argument("checkpoint numeric parameter slot/group layout does not match target");

    if (optimizer) {
        if (!saved_optimizer)
            throw std::invalid_argument("target optimizer was supplied but checkpoint has no optimizer state");
        if (optimizer_json != expected_optimizer_layout)
            throw std::invalid_argument("checkpoint optimizer parameter groups do not match numeric slots");
    }

    torch::serialize::InputArchive slot_archive;
    archive.read("parameters", slot_archive);
    std::set<std::string> expected_slot_keys;
    for (const auto id : snapshot.roots) expected_slot_keys.insert(slot_archive_key(id));
    require_keys(slot_archive, expected_slot_keys, "parameter-slot archive");

    std::vector<std::pair<at::Tensor, at::Tensor>> staged;
    staged.reserve(snapshot.roots.size());
    for (const auto id : snapshot.roots) {
        auto target = parameters.get(id);
        at::Tensor value;
        slot_archive.read(slot_archive_key(id), value, false);
        if (!value.defined() || value.sizes() != target.sizes() ||
            value.scalar_type() != target.scalar_type())
            throw std::invalid_argument("checkpoint parameter slot has a mismatched shape or dtype");
        auto staged_value = value.to(target.device()).contiguous();
        staged.emplace_back(target, std::move(staged_value));
    }

    torch::serialize::InputArchive optimizer_archive;
    if (optimizer) archive.read("optimizer", optimizer_archive);

    std::vector<std::pair<at::Tensor, at::Tensor>> backups;
    backups.reserve(staged.size());
    for (const auto& entry : staged)
        backups.emplace_back(entry.first, entry.first.detach().clone());

    std::string optimizer_backup_bytes;
    if (optimizer) {
        torch::serialize::OutputArchive backup_archive;
        optimizer->save(backup_archive);
        std::ostringstream backup_stream(std::ios::out | std::ios::binary);
        backup_archive.save_to(backup_stream);
        optimizer_backup_bytes = backup_stream.str();
    }

    try {
        copy_values(staged);
        if (optimizer) optimizer->load(optimizer_archive);
    } catch (...) {
        try {
            copy_values(backups);
        } catch (...) {
            // Preserve the original checkpoint failure; all parameter values
            // were staged before mutation, and rollback is best-effort if the
            // underlying device itself has failed.
        }
        if (optimizer && !optimizer_backup_bytes.empty()) {
            try {
                std::istringstream backup_stream(optimizer_backup_bytes,
                                                 std::ios::in | std::ios::binary);
                torch::serialize::InputArchive backup_archive;
                backup_archive.load_from(backup_stream);
                optimizer->load(backup_archive);
            } catch (...) {
                // The primary failure remains the useful diagnostic. Torch's
                // optimizer load contract does not expose transactional state.
            }
        }
        throw;
    }
}

void load_checkpoint(
    const std::filesystem::path& path,
    const ParameterGroup& group,
    const BiologicalManifest& expected_manifest,
    torch::optim::Optimizer* optimizer) {
    auto owner = selected_group_parameters(group);
    load_checkpoint(path, owner, expected_manifest, optimizer);
}

} // namespace cellerator::nn
