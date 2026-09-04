#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::backend::nvptx {

enum class direct_ptx_type_v1 : std::uint8_t {
    predicate = 1u,
    b32,
    b64,
    u32,
    u64,
    s32,
    s64,
    f32,
    f64,
};

enum class direct_ptx_address_space_v1 : std::uint8_t {
    none = 0u,
    generic,
    global,
    shared,
    local,
    constant,
    parameter,
};

enum direct_ptx_memory_effect_v1 : std::uint16_t {
    direct_ptx_memory_none_v1 = 0u,
    direct_ptx_memory_read_v1 = 1u << 0u,
    direct_ptx_memory_write_v1 = 1u << 1u,
    direct_ptx_memory_atomic_v1 = 1u << 2u,
    direct_ptx_memory_volatile_v1 = 1u << 3u,
};

enum class direct_ptx_node_kind_v1 : std::uint8_t {
    label = 1u,
    instruction,
    barrier,
    collective,
};

enum class direct_ptx_barrier_kind_v1 : std::uint8_t {
    none = 0u,
    synchronize,
    arrive,
};

enum class direct_ptx_collective_kind_v1 : std::uint8_t {
    none = 0u,
    vote_all,
    vote_any,
    shuffle,
    reduction,
};

enum class direct_ptx_collective_scope_v1 : std::uint8_t {
    none = 0u,
    warp,
    cooperative_thread_array,
    cluster,
};

struct direct_ptx_register_v1 {
    std::uint32_t identity = 0u;
    direct_ptx_type_v1 type = direct_ptx_type_v1::b32;
};

struct direct_ptx_parameter_v1 {
    std::string name;
    direct_ptx_type_v1 type = direct_ptx_type_v1::b64;
    direct_ptx_address_space_v1 address_space = direct_ptx_address_space_v1::parameter;
    std::uint32_t alignment = 0u;
};

struct direct_ptx_instruction_requirement_v1 {
    std::uint16_t minimum_sm_major = 0u;
    std::uint16_t minimum_sm_minor = 0u;
    std::string feature;
};

// A typed extension node attached to a Cellerator Realization IR identity.
// Register references are stable numeric identities rather than printed PTX names.
struct direct_ptx_operation_v1 {
    std::uint64_t realization_node_identity = 0u;
    direct_ptx_node_kind_v1 kind = direct_ptx_node_kind_v1::instruction;
    std::string opcode;
    std::uint32_t result_register = 0u;
    std::vector<std::uint32_t> operand_registers;
    std::uint32_t predicate_register = 0u;
    bool predicate_negated = false;
    std::string label;
    direct_ptx_address_space_v1 address_space = direct_ptx_address_space_v1::none;
    std::uint16_t memory_effects = direct_ptx_memory_none_v1;
    direct_ptx_barrier_kind_v1 barrier = direct_ptx_barrier_kind_v1::none;
    direct_ptx_collective_kind_v1 collective = direct_ptx_collective_kind_v1::none;
    direct_ptx_collective_scope_v1 collective_scope = direct_ptx_collective_scope_v1::none;
    std::uint16_t collective_threads = 0u;
    direct_ptx_instruction_requirement_v1 requirement;
};

struct direct_ptx_kernel_v1 {
    std::uint64_t realization_kernel_identity = 0u;
    std::string symbol;
    std::vector<direct_ptx_parameter_v1> parameters;
    std::vector<direct_ptx_register_v1> registers;
    std::vector<direct_ptx_operation_v1> operations;
};

enum class direct_ptx_model_status_v1 : std::uint8_t {
    success = 0u,
    invalid_kernel,
    invalid_parameter,
    invalid_register,
    invalid_operation,
    invalid_reference,
    unsupported_requirement,
    parse_error,
};

[[nodiscard]] direct_ptx_model_status_v1 validate_direct_ptx_kernel_v1(
    const direct_ptx_kernel_v1& kernel,
    std::uint16_t target_sm_major,
    std::uint16_t target_sm_minor,
    std::string* diagnostic = nullptr) noexcept;

[[nodiscard]] std::string print_direct_ptx_kernel_model_v1(
    const direct_ptx_kernel_v1& kernel);

[[nodiscard]] direct_ptx_model_status_v1 parse_direct_ptx_kernel_model_v1(
    std::string_view text,
    direct_ptx_kernel_v1* kernel,
    std::string* diagnostic = nullptr);

}  // namespace Cellerator::compiler::backend::nvptx
