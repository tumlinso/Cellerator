#pragma once

#include <Cellerator/compute/operation/indexed_mechanism/incidence.hh>
#include <Cellerator/execution/program/program_v2.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace cellerator::compute::operation::indexed {

// This deliberately small vocabulary is an executable implementation, rather
// than a symbolic language or a per-edge callback interface.
enum class evaluator_opcode : std::uint8_t {
    sum,
    product,
    first_minus_product_tail
};

enum class evaluation_status : std::uint8_t {
    success, invalid_block, duplicate_block, invalid_arity, invalid_binding,
    invalid_effect, cuda_error
};

struct registered_block {
    identity evaluator{};
    evaluator_opcode opcode = evaluator_opcode::sum;
    std::uint64_t minimum_arguments = 0;
    // The declaration is intentional: a custom name alone is not executable.
    bool dependencies_known = true;
    bool writes_only_declared_outputs = true;
    bool deterministic = true;
};

class block_registry {
public:
    evaluation_status register_block(registered_block block) noexcept;
    const registered_block* find(identity evaluator) const noexcept;
private:
    std::vector<registered_block> blocks_;
};

// Computes one declared output.  A false predicate excludes the whole block:
// it does not read arguments, sanitize nonfinites, or overwrite destination.
evaluation_status evaluate_f32(const registered_block& block, bool predicate,
                               std::span<const float> arguments,
                               std::span<float> destination) noexcept;
evaluation_status evaluate_f16(const registered_block& block, bool predicate,
                               std::span<const std::uint16_t> arguments,
                               std::span<std::uint16_t> destination) noexcept;

// CUDA entry points use FP32 accumulation for both FP32 and IEEE half input.
// `stream` is a cudaStream_t passed opaquely to avoid a CUDA header dependency
// in ordinary consumers.  Each launch uses O(1) scratch.
evaluation_status evaluate_cuda_f32(const registered_block& block, bool predicate,
                                    const float* arguments, std::size_t argument_count,
                                    float* destination, void* stream = nullptr) noexcept;
evaluation_status evaluate_cuda_f16(const registered_block& block, bool predicate,
                                    const std::uint16_t* arguments, std::size_t argument_count,
                                    std::uint16_t* destination, void* stream = nullptr) noexcept;

// A caller owns this immutable state for the full prepared-program lifetime.
// Its input view is packed device FP32 arguments (including any supplied
// forcing values).  The generic CE program owns stage ordering and buffers.
struct prepared_evaluator_stage {
    registered_block block{};
    std::uint64_t argument_count = 0;
    std::uint64_t required_value_generation = 0;
};
struct evaluator_stage_values {
    static constexpr std::uint64_t magic = 0x4e4631414d454348ULL; // NF1AMECH
    std::uint64_t binding_magic = magic;
    std::uint64_t value_generation = 0;
    bool predicate = true;
};
bool valid_prepared_stage(const prepared_evaluator_stage& stage) noexcept;
execution::program::program_status admit_evaluator_stage(
    const void* prepared_state, const execution::program::launch_binding_v2& binding,
    void* caller_stream) noexcept;
execution::program::program_status launch_evaluator_stage(
    const void* prepared_state, const execution::program::launch_binding_v2& binding,
    void* caller_stream) noexcept;
execution::program::prepared_stage_v2 make_prepared_stage(
    std::uint64_t stable_stage_id, std::uint64_t candidate_id,
    const prepared_evaluator_stage& state, std::uint32_t binding_index,
    std::uint64_t first_dependency = 0, std::uint32_t dependency_count = 0) noexcept;

} // namespace cellerator::compute::operation::indexed
