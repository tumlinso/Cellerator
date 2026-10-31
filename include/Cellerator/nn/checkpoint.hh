#pragma once

#include <Cellerator/nn/manifest.hh>
#include <Cellerator/nn/parameters.hh>

#include <filesystem>

namespace torch::optim {
class Optimizer;
} // namespace torch::optim

namespace cellerator::nn {

inline constexpr char native_checkpoint_format[] = "cellerator.nn.parameter-checkpoint";
inline constexpr int native_checkpoint_version = 1;

// Save numeric parameter slots and their structural layout in an ordinary
// Torch archive. If supplied, the optimizer uses its standard Torch archive
// serializer and is bound to explicit numeric slot IDs. The manifest is stored
// as the shared v1 biological JSON contract with weights=null (the archive
// contains the weights, so a self-referential file digest is not meaningful).
void save_checkpoint(
    const std::filesystem::path& path,
    const Parameters& parameters,
    const BiologicalManifest& manifest,
    torch::optim::Optimizer* optimizer = nullptr);

// Save one explicit parameter group as a local numeric-slot checkpoint. The
// selected group's tensors are adopted by a temporary metadata owner without
// moving, cloning, or detaching them; tied leaves within the selection remain
// tied in the archive. Optimizers must contain selected leaves only.
void save_checkpoint(
    const std::filesystem::path& path,
    const ParameterGroup& group,
    const BiologicalManifest& manifest,
    torch::optim::Optimizer* optimizer = nullptr);

// Validate the complete biological declaration and exact numeric owner/group/
// slot layout, then stage every slot tensor before mutating any parameter.
// Tied slots are copied through their canonical slot and remain the same leaf.
// An optimizer is restored only if one was saved, and its parameter groups must
// name the same numeric slots in the same order.
void load_checkpoint(
    const std::filesystem::path& path,
    Parameters& parameters,
    const BiologicalManifest& expected_manifest,
    torch::optim::Optimizer* optimizer = nullptr);

// Restore one explicit parameter group. Validation completes against a local
// slot layout before the adopted shared leaves are copied, so unrelated owner
// groups remain untouched (except when they intentionally share a selected
// tied leaf).
void load_checkpoint(
    const std::filesystem::path& path,
    const ParameterGroup& group,
    const BiologicalManifest& expected_manifest,
    torch::optim::Optimizer* optimizer = nullptr);

} // namespace cellerator::nn
