#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
namespace cellerator::compiler::api::v1 {
enum class ceir_level_v1 : std::uint8_t { semantic=0, planning, realization };
enum class ceir_validation_v1 : std::uint8_t { checked=0, trusted, unsafe_mode };
struct ceir_module_v1 { ceir_level_v1 level{}; std::string text; std::vector<std::string> provenance; };
using ceir_snapshot_v1 = std::shared_ptr<const ceir_module_v1>;
class ceir_builder_v1 {
public:
 explicit ceir_builder_v1(ceir_level_v1 level) : module_{level,{},{}} {}
 void set_text(std::string text) { module_.text=std::move(text); }
 void add_provenance(std::string value) { module_.provenance.push_back(std::move(value)); }
 [[nodiscard]] ceir_snapshot_v1 freeze() const { return std::make_shared<const ceir_module_v1>(module_); }
private: ceir_module_v1 module_;
};
[[nodiscard]] ceir_snapshot_v1 parse_ceir_v1(ceir_level_v1 level, std::string text);
[[nodiscard]] std::string print_ceir_v1(const ceir_snapshot_v1& module);
[[nodiscard]] ceir_builder_v1 clone_ceir_v1(const ceir_snapshot_v1& module);
[[nodiscard]] bool validate_ceir_v1(const ceir_snapshot_v1& module, ceir_validation_v1 mode) noexcept;
[[nodiscard]] std::vector<std::uint8_t> serialize_ceir_v1(const ceir_snapshot_v1& module);
}
