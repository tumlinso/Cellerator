#include <Cellerator/memory/generation_marks.hh>

#include <algorithm>
#include <limits>

namespace cellerator::memory {

status advance_generation(generation_marks *table) noexcept {
    if (table == nullptr || table->generation == 0u)
        return status::invalid_argument;
    if (table->generation == std::numeric_limits<std::uint32_t>::max())
        return status::generation_clear_required;
    ++table->generation;
    return status::success;
}

status reset_generation_marks_host(generation_marks *table) noexcept {
    if (table == nullptr || (table->count != 0u && table->marks == nullptr))
        return status::invalid_argument;
    switch (table->where.kind) {
    case domain::host:
    case domain::host_numa:
    case domain::host_pinned:
    case domain::host_pinned_write_combined:
        break;
    default:
        return status::invalid_placement;
    }
    if (table->count != 0u)
        std::fill(table->marks, table->marks + table->count, 0u);
    table->generation = 1u;
    return status::success;
}

} // namespace cellerator::memory
