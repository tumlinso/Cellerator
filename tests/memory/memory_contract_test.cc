#include <Cellerator/memory/allocation.hh>
#include <Cellerator/memory/compiler_hints.hh>
#include <Cellerator/memory/generation_marks.hh>
#include <Cellerator/memory/image.hh>
#include <Cellerator/memory/view.hh>
#include <Cellerator/memory/workspace.hh>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits>

namespace {

namespace memory = cellerator::memory;

int require(bool condition, const char *message) {
    if (condition) return 0;
    std::cerr << "memory contract test failed: " << message << '\n';
    return 1;
}

CELLERATOR_NOINLINE int hinted_sum(
    const int *CELLERATOR_RESTRICT values,
    std::size_t count) {
    CELLERATOR_ASSUME(values != nullptr);
    const int *aligned = static_cast<const int *>(
        CELLERATOR_ASSUME_ALIGNED(values, alignof(int)));
    int result = 0;
    for (std::size_t index = 0; index < count; ++index)
        if (CELLERATOR_LIKELY(aligned[index] >= 0)) result += aligned[index];
    return result;
}

} // namespace

int main() {
    static_assert(std::is_standard_layout<memory::allocation>::value,
        "allocation record must be standard layout");
    static_assert(std::is_trivially_copyable<memory::const_array_view<int>>::value,
        "views must be trivially copyable");

    memory::allocation_request host_request{};
    host_request.bytes = 128u;
    host_request.where = {memory::domain::host, -1, -1, 0u};
    if (require(memory::validate_allocation_request(host_request)
                    == memory::status::success,
                "valid host allocation request")
        || require(memory::validate_allocation_request(
                       {128u, 3u, host_request.where})
                       == memory::status::invalid_alignment,
                   "non-power-of-two alignment rejected")
        || require(memory::validate_allocation_request(
                       {128u, 64u, {memory::domain::device, -1, -1, 0u}})
                       == memory::status::invalid_placement,
                   "device ordinal required")) return 1;

    alignas(64) unsigned char external[256]{};
    memory::allocation external_record{};
    if (require(memory::bind_external_allocation(
                    external, sizeof(external), 64u, 9u, &external_record)
                    == memory::status::success,
                "external allocation bind")
        || require(external_record.where.kind == memory::domain::external
                       && external_record.generation == 9u,
                   "external allocation metadata")
        || require(memory::bind_external_allocation(
                       external + 1u, 8u, 64u, 1u, &external_record)
                       == memory::status::invalid_alignment,
                   "misaligned external bind rejected")) return 1;

    unsigned char unaligned_storage[193]{};
    memory::workspace workspace{
        unaligned_storage + 1u, 192u, 0u,
        {memory::domain::host, -1, -1, 0u}};
    std::uint32_t *first = nullptr;
    void *second = nullptr;
    if (require(memory::take(&workspace, 7u, 64u, &first)
                    == memory::status::success,
                "aligned typed take")
        || require((reinterpret_cast<std::uintptr_t>(first) & 63u) == 0u,
                   "absolute address alignment")
        || require(memory::take_bytes(&workspace, 16u, 16u, &second)
                       == memory::status::success,
                   "second workspace take")
        || require(reinterpret_cast<std::uintptr_t>(second)
                       >= reinterpret_cast<std::uintptr_t>(first + 7u),
                   "workspace slices do not overlap")) return 1;

    const std::size_t cursor_before_failure = workspace.cursor;
    void *failure = reinterpret_cast<void *>(std::uintptr_t{1});
    std::uint64_t *typed_failure = reinterpret_cast<std::uint64_t *>(
        std::uintptr_t{1});
    if (require(memory::take_bytes(&workspace, 4096u, 8u, &failure)
                    == memory::status::capacity_exceeded,
                "workspace capacity rejection")
        || require(failure == nullptr && workspace.cursor == cursor_before_failure,
                   "workspace failure is atomic")
        || require(memory::take_bytes(&workspace, 1u, 6u, &failure)
                       == memory::status::invalid_alignment,
                   "workspace alignment rejection")
        || require(memory::take<std::uint64_t>(&workspace,
                       std::numeric_limits<std::size_t>::max(), &typed_failure)
                       == memory::status::arithmetic_overflow,
                   "typed workspace multiplication overflow")
        || require(memory::reset(&workspace) == memory::status::success
                       && workspace.cursor == 0u,
                   "workspace reset preserves storage")) return 1;

    alignas(64) unsigned char image_storage[256]{};
    memory::image_header header{};
    header.magic = 0x4d454d31u;
    header.schema_version = 1u;
    header.total_bytes = sizeof(image_storage);
    header.required_alignment = 64u;
    const memory::const_image_view image{
        image_storage, sizeof(image_storage),
        {memory::domain::host, -1, -1, 0u}};
    const void *resolved = nullptr;
    if (require(memory::validate_image_header(
                    header, image, 0x4d454d31u, 1u)
                    == memory::status::success,
                "image header validation")
        || require(memory::resolve_image_span(image, 64u, 32u, 32u, &resolved)
                       == memory::status::success
                       && resolved == image_storage + 64u,
                   "relative span resolution")
        || require(memory::validate_image_span(
                       sizeof(image_storage), memory::rel32{240u}, 32u, 8u)
                       == memory::status::capacity_exceeded,
                   "relative span range rejection")
        || require(memory::validate_image_span(
                       sizeof(image_storage), memory::rel64{65u}, 1u, 64u)
                       == memory::status::invalid_alignment,
                   "relative span alignment rejection")) return 1;

    std::uint32_t marks[4]{0u, 0u, 0u, 0u};
    memory::generation_marks table{
        marks, 4u, 1u, {memory::domain::host, -1, -1, 0u}};
    if (require(memory::insert(&table, 2u) == memory::status::success
                    && memory::contains(table, 2u),
                "generation mark insertion")
        || require(memory::advance_generation(&table) == memory::status::success
                       && !memory::contains(table, 2u),
                   "generation advance is a logical clear")) return 1;
    table.generation = std::numeric_limits<std::uint32_t>::max();
    if (require(memory::advance_generation(&table)
                    == memory::status::generation_clear_required,
                "generation wrap requires explicit clear")
        || require(memory::reset_generation_marks_host(&table)
                       == memory::status::success
                       && table.generation == 1u && marks[2] == 0u,
                   "host generation wrap clear")) return 1;

    int values[3]{1, 2, 3};
    memory::array_view<int> mutable_view{
        values, 3u, {memory::domain::host, -1, -1, 0u}};
    const memory::const_array_view<int> const_view = memory::as_const(mutable_view);
    if (require(const_view.data == values && hinted_sum(values, 3u) == 6,
                "const view and compiler hints")) return 1;

    std::cout << "memory substrate host contract passed\n";
    return 0;
}
