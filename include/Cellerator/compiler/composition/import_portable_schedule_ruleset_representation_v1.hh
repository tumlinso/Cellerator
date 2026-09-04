#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace Cellerator::compiler::composition {
enum class replay_mode_v1:std::uint8_t{exact,compatible,replan};
struct portable_schedule_v1{std::vector<std::string> operation_order,atom_requirements,partial_tree,canonical_recovery;replay_mode_v1 replay=replay_mode_v1::exact;};
[[nodiscard]] std::uint64_t portable_schedule_identity_v1(const portable_schedule_v1&);
[[nodiscard]] bool validate_portable_schedule_v1(const portable_schedule_v1&,std::string*error=nullptr);
} // namespace Cellerator::compiler::composition
