#pragma once
#include <cstdint>
#include <string_view>
namespace cellerator::compiler::diagnostics::v1 {enum class planning_barrier:std::uint8_t{opaque_cxx_call=0,field_boundary,unknown_extension,effect,alias_uncertainty,profile_widening,hard_constraint};struct source_range{std::uint64_t file=0;std::uint32_t begin=0,end=0;};struct barrier_report{planning_barrier barrier=planning_barrier::opaque_cxx_call;source_range range{};std::string_view explanation;bool valid=false;};[[nodiscard]] barrier_report explain_planning_barrier(planning_barrier,source_range) noexcept;}
