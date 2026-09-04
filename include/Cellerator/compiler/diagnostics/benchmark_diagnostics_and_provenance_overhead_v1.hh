#pragma once
#include <cstdint>
#include <vector>
namespace cellerator::compiler::diagnostics::v1 {enum class provenance_level:std::uint8_t{disabled=0,minimal,full};enum class translation_unit_size:std::uint8_t{small=0,large};struct provenance_measurement{provenance_level level=provenance_level::disabled;translation_unit_size size=translation_unit_size::small;std::uint64_t compile_ns=0,peak_rss_bytes=0,object_bytes=0,hot_runtime_bytes=0;};struct provenance_budget{std::uint32_t compile_percent=25,rss_percent=20,object_percent=30;};[[nodiscard]] bool provenance_overhead_within_budget(const std::vector<provenance_measurement>&,provenance_budget) noexcept;}
