#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compiler::pass::v1 {

struct realization_object_v1 { std::uint64_t id = 0; std::string kind; };
struct realization_stage_v1 {
    std::uint64_t id = 0;
    std::string operation;
    std::vector<std::uint64_t> dependencies;
};
struct realization_binding_v1 {
    std::uint64_t stage = 0;
    std::uint64_t object = 0;
};
struct realization_native_fragment_v1 {
    std::uint64_t stage = 0;
    std::string provider;
    std::vector<std::uint8_t> bytes;
};

struct realization_pass_context_v1 {
    std::vector<realization_object_v1>* physical_covers = nullptr;
    std::vector<realization_object_v1>* projections = nullptr;
    std::vector<realization_object_v1>* packs = nullptr;
    std::vector<realization_stage_v1>* stages = nullptr;
    std::vector<realization_binding_v1>* bindings = nullptr;
    std::vector<realization_object_v1>* target_operations = nullptr;
    std::vector<realization_native_fragment_v1>* native_fragments = nullptr;
    std::vector<std::string>* diagnostics = nullptr;
};

using realization_pass_run_v1 = bool (*)(realization_pass_context_v1&) noexcept;
enum class realization_pass_status_v1 : std::uint8_t {
    success = 0, invalid_context, pass_failed, invalid_result,
};

[[nodiscard]] realization_pass_status_v1 run_custom_realization_pass_v1(
    realization_pass_context_v1& context, realization_pass_run_v1 pass) noexcept;

}  // namespace cellerator::compiler::pass::v1
