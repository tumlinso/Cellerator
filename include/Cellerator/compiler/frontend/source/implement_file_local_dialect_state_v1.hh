#pragma once

#include <Cellerator/compiler/frontend/source/define_the_unified_source_location_model_v1.hh>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace Cellerator::compiler::frontend::source {

struct dialect_file_state_v1 {
    source_space_id_v1 include_instance = invalid_source_space_v1;
    std::optional<std::uint64_t> activation_offset;
    std::string revision;
};

class file_local_dialect_stack_v1 {
  public:
    [[nodiscard]] bool enter_file(source_space_id_v1 include_instance);
    [[nodiscard]] bool leave_file(source_space_id_v1 include_instance);
    [[nodiscard]] bool activate(source_location_v1 location, std::string revision);
    [[nodiscard]] bool active_at(source_location_v1 location) const noexcept;
    [[nodiscard]] const dialect_file_state_v1* current() const noexcept;
    [[nodiscard]] std::size_t depth() const noexcept { return stack_.size(); }

  private:
    std::vector<dialect_file_state_v1> stack_;
};

} // namespace Cellerator::compiler::frontend::source
