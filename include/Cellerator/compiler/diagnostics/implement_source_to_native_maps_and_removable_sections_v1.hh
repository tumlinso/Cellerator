#pragma once
#include <cstdint>
#include <vector>
namespace cellerator::compiler::diagnostics::v1 {
enum class provenance_storage:std::uint8_t{sidecar=0,object_debug_section,separate_debug_file};
struct source_native_entry{std::uint64_t source_id=0,native_symbol=0,native_offset=0;};
struct provenance_image{std::uint64_t hot_bytes=0;provenance_storage storage=provenance_storage::sidecar;std::vector<source_native_entry> cold_map;};
[[nodiscard]] bool valid_source_native_map(const provenance_image&) noexcept;
[[nodiscard]] provenance_image strip_provenance(provenance_image) noexcept;
}
