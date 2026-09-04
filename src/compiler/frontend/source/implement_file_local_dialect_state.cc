#include <Cellerator/compiler/frontend/source/implement_file_local_dialect_state_v1.hh>

#include <utility>

namespace Cellerator::compiler::frontend::source {

bool file_local_dialect_stack_v1::enter_file(source_space_id_v1 include_instance) {
    if (include_instance == invalid_source_space_v1) {
        return false;
    }
    stack_.push_back({include_instance, std::nullopt, {}});
    return true;
}

bool file_local_dialect_stack_v1::leave_file(source_space_id_v1 include_instance) {
    if (stack_.empty() || stack_.back().include_instance != include_instance) {
        return false;
    }
    stack_.pop_back();
    return true;
}

bool file_local_dialect_stack_v1::activate(source_location_v1 location, std::string revision) {
    if (stack_.empty() || stack_.back().include_instance != location.space ||
        stack_.back().activation_offset.has_value()) {
        return false;
    }
    stack_.back().activation_offset = location.byte_offset;
    stack_.back().revision = std::move(revision);
    return true;
}

bool file_local_dialect_stack_v1::active_at(source_location_v1 location) const noexcept {
    if (stack_.empty()) {
        return false;
    }
    const auto& state = stack_.back();
    return state.include_instance == location.space && state.activation_offset &&
           location.byte_offset >= *state.activation_offset;
}

const dialect_file_state_v1* file_local_dialect_stack_v1::current() const noexcept {
    return stack_.empty() ? nullptr : &stack_.back();
}

} // namespace Cellerator::compiler::frontend::source
