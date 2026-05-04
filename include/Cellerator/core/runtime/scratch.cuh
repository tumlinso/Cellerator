#pragma once

#include <cstddef>

namespace cellerator::core::runtime {

struct scratch_arena {
    void *data = nullptr;
    std::size_t bytes = 0;
};

void init(scratch_arena *arena);
void clear(scratch_arena *arena);
void *request_scratch(scratch_arena *arena, std::size_t bytes);

} // namespace cellerator::core::runtime
