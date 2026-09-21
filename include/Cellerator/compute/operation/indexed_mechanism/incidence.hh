#pragma once
#include <Cellerator/compute/operation/native_foundation_contract.hh>
#include <span>
#include <vector>
namespace cellerator::compute::operation::indexed {
using identity=nf1::identity;
enum class incidence_status { success, invalid_identity, invalid_axis, invalid_slot,
    invalid_index, duplicate_writer, invalid_effect, invalid_binding, allocation_failure };
struct indexed_axis { execution::persistent_axis_identity identity{};std::uint64_t extent=0; };
struct argument_index {
    std::uint64_t slot=0;identity role{};std::uint64_t axis=0,index=0;
};
struct output_index {
    std::uint64_t slot=0;identity role{};std::uint64_t axis=0,index=0;
    identity assembly_owner{};
    execution::output_effect_contract effect{execution::output_update_kind::overwrite,false,false,0,
        execution::invalid_scalar_binding_id,execution::invalid_scalar_binding_id};
};
struct mechanism_incidence {
    // Execution incidence identity, never an inferred biological/latent entity.
    identity instance{}, evaluator{};
    std::span<const argument_index> arguments{};
    std::span<const output_index> outputs{};
};
struct prepared_incidence {
    identity instance{}, evaluator{};
    std::vector<argument_index> arguments; // indexed by semantic slot, not source index
    std::vector<output_index> outputs;
};
struct host_axis_f64 { execution::persistent_axis_identity identity{};std::span<const double> values{}; };
class argument_incidence {
public:
    // Copies bounded metadata, reconstructing logical slots from any physical
    // record order. Multiplicity is retained; no commutativity is inferred.
    incidence_status prepare(std::span<const indexed_axis> inputs,
        std::span<const indexed_axis> outputs,std::span<const mechanism_incidence>) noexcept;
    // Synchronous scalar host gathering only; H03 adds batched device lowering.
    // Rejects every mismatch/alias before writing any gathered argument.
    incidence_status gather_f64(std::uint64_t instance,std::span<const host_axis_f64>,
        std::span<double> arguments) const noexcept;
    const std::vector<prepared_incidence>& mechanisms() const noexcept { return mechanisms_; }
    const std::vector<indexed_axis>& inputs() const noexcept { return inputs_; }
    const std::vector<indexed_axis>& outputs() const noexcept { return outputs_; }
private:
    std::vector<indexed_axis> inputs_,outputs_;
    std::vector<prepared_incidence> mechanisms_;
};
} // namespace cellerator::compute::operation::indexed
