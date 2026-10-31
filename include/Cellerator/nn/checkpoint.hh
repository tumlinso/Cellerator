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

} // namespace cellerator::nn
