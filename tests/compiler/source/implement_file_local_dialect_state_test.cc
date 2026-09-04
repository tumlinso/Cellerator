#include <Cellerator/compiler/frontend/source/implement_file_local_dialect_state_v1.hh>

#include <iostream>
#include <stdexcept>

using namespace Cellerator::compiler::frontend::source;

int main() {
    try {
        file_local_dialect_stack_v1 state;
        if (!state.enter_file(1) || !state.activate({1, 10}, "0.1") ||
            state.active_at({1, 9}) || !state.active_at({1, 10})) {
            throw std::runtime_error("includer activation boundary failed");
        }
        if (!state.enter_file(2) || state.active_at({2, 20})) {
            throw std::runtime_error("activation leaked into inactive header");
        }
        if (!state.activate({2, 21}, "0.1") || !state.active_at({2, 22}) ||
            !state.leave_file(2) || !state.active_at({1, 100})) {
            throw std::runtime_error("activated header did not restore includer state");
        }
        // A repeated include has a distinct FileID and starts inactive, even if
        // the previous instance was active or an include guard skips its body.
        if (!state.enter_file(3) || state.active_at({3, 0}) || !state.leave_file(3) ||
            !state.leave_file(1) || state.depth() != 0) {
            throw std::runtime_error("repeated include instance leaked state");
        }
        std::cout << "validated file-local dialect state over nested include instances\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
