#pragma once

#include <Cellerator/execution/atom_fragment/prepared_atom_fragment_v1.hh>
#include <Cellerator/execution/dependency/distributed.hh>
#include <Cellerator/execution/joint_compiler/atom_requirement_v1.hh>
#include <Cellerator/ops/mechanism/gated_product_binding.hh>
#include <Cellerator/ops/mechanism/packed_gated_product.cuh>
#include <Cellerator/tensor/block_masked/owner.hh>

#include <memory>
#include <span>
#include <vector>

namespace cellerator::execution::dependency {

// Identities are assigned by the caller's persistent identity authority.
// In particular, they are neither device ordinals nor runtime addresses.
struct PackedAtomIdentities {
    joint_compiler::persistent_identity_v1 atom, requirement, coverage, affordance, binding;
};

struct PackedProgramPreparation {
    std::shared_ptr<const ops::mechanism::PreparedGatedProductBinding> semantics;
    std::shared_ptr<const tensor::block_masked::DeviceStructure> structure;
    std::span<const std::uint64_t> group_indices;
    PackedAtomIdentities identities{};
    structure::structure_id structure_identity{};
    structure::order_id physical_order{};
    std::uint64_t structure_epoch = 0;
    std::uint64_t placement_revision = 0;
    // Pointer identities for prepared, device-resident topology and maps.
    // The owner must have uploaded exactly the supplied semantic groups, in
    // group_indices order, and the maps prepared from this Structure; this
    // host-only boundary cannot inspect device memory. These uploads must stay
    // immutable for the lifetime of this program. Numerical source pointers
    // are supplied on every launch.
    ops::mechanism::PackedGatedProductInput<std::uint16_t> prepared_input{};
};

struct PackedProgramLaunch {
    const geometry::dependency::SourceDescriptor* source = nullptr;
    DeviceExtentBinding state{};
    ExtentRequirement expected{};
    const tensor::block_masked::RetainedRead<std::uint16_t>* read = nullptr;
    const memory::cuda::Event* readiness = nullptr;
    const memory::cuda::Stream* stream = nullptr;
    ops::mechanism::PackedGatedProductInput<std::uint16_t> input{};
    ops::mechanism::PackedGatedProductCapacities capacities{};
    float* output = nullptr;
    const float* output_gradient = nullptr;
    ops::mechanism::PackedGatedProductGradients gradients{};
    ops::mechanism::PackedGatedProductSaved saved{};
    std::uint32_t* status = nullptr;
};

// Pins preparation metadata, not runtime data. Callers retain read handles
// through the final CUDA completion, including backward. z and parameters have
// independent source/phase/readiness contracts at the distributed owner.
class PreparedPackedProgram final {
public:
    explicit PreparedPackedProgram(const PackedProgramPreparation&);
    ~PreparedPackedProgram();
    PreparedPackedProgram(const PreparedPackedProgram&) = delete;
    PreparedPackedProgram& operator=(const PreparedPackedProgram&) = delete;
    PreparedPackedProgram(PreparedPackedProgram&&) = delete;
    PreparedPackedProgram& operator=(PreparedPackedProgram&&) = delete;

    program::program_status admit_forward(const PackedProgramLaunch&) const noexcept;
    program::program_status admit_backward(const PackedProgramLaunch&) const noexcept;
    program::program_status forward(const PackedProgramLaunch&) const noexcept;
    program::program_status backward(const PackedProgramLaunch&) const noexcept;
    const joint_compiler::atom_requirement_v1& requirement() const noexcept;
    const atom_fragment::prepared_atom_fragment_v1& forward_fragment() const noexcept;
    const atom_fragment::prepared_atom_fragment_v1& backward_fragment() const noexcept;
    std::span<const std::uint64_t> coverage_coordinates() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct PackedProgramInvocation {
    const PreparedPackedProgram* program = nullptr;
    const PackedProgramLaunch* launch = nullptr;
};

// Admit every device before any program callback is dispatched. This is a
// bounded multi-device submission boundary, not a graph or scheduler. As with
// program_v2, failures after dispatch cannot roll back earlier submissions.
program::program_status admit_packed_programs(
    std::span<const PackedProgramInvocation> invocations, bool backward) noexcept;
program::program_status execute_packed_programs(
    std::span<const PackedProgramInvocation> invocations, bool backward) noexcept;

} // namespace cellerator::execution::dependency
