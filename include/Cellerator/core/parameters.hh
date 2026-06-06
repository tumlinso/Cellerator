#pragma once

#include <cstddef>
#include <cstdint>

namespace cellerator::core {

enum class parameter_scalar_type : std::uint8_t {
    unknown = 0,
    f16,
    bf16,
    f32,
    f64,
    i32,
    i64,
    u8
};

enum class parameter_memory_space : std::uint8_t {
    host = 0,
    device
};

enum class parameter_role : std::uint8_t {
    learned = 0,
    optimizer_state
};

constexpr std::size_t parameter_max_rank = 8;

struct parameter_descriptor {
    const char *name = nullptr;
    parameter_scalar_type scalar_type = parameter_scalar_type::unknown;
    parameter_memory_space memory_space = parameter_memory_space::device;
    int device_ordinal = -1;
    void *data = nullptr;
    std::uint8_t rank = 0;
    std::int64_t shape[parameter_max_rank] = {};
    std::int64_t stride[parameter_max_rank] = {};
    bool writable = false;
    parameter_role role = parameter_role::learned;
};

struct parameter_view {
    const parameter_descriptor *parameters = nullptr;
    std::size_t count = 0;
};

} // namespace cellerator::core
