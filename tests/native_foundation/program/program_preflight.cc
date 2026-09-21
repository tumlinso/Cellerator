#include <Cellerator/execution/program/program_v2.h>
#include <stdexcept>

namespace p = cellerator::execution::program;
namespace {
int launches = 0;
p::program_status launch(const void*, const p::launch_binding_v2&, void*) noexcept {
    ++launches;
    return p::program_status::success;
}
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
}
int main() {
    p::prepared_stage_v2 stages[] = {
        {1, 1, nullptr, launch, 0, 0, 0, 0},
        {2, 1, nullptr, launch, 0, 0, 1, 64},
    };
    p::prepared_program_v2 program{2, 0, stages, 2, nullptr, 0};
    p::launch_binding_v2 bindings[2]{};
    require(p::execute_prepared_program_v2(program, bindings, 2, nullptr) ==
                p::program_status::insufficient_bindings,
            "late invalid binding must reject");
    require(launches == 0, "preflight rejection must submit no earlier stage");
    bindings[1].workspace_bytes = 64;
    require(p::execute_prepared_program_v2(program, bindings, 2, nullptr) ==
                p::program_status::success,
            "valid program should execute");
    require(launches == 2, "both accepted stages should launch");
}
