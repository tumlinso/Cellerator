#include <Cellerator/compiler/backend/benchmark_cpu_backend_complete_cost_v1.hh>

#include <cassert>

namespace cb = cellerator::compiler::backend::v1;

namespace {
struct state { volatile std::uint64_t value = 0; };
bool compile(void* context) noexcept { static_cast<state*>(context)->value += 1; return true; }
bool prepare(void* context) noexcept { static_cast<state*>(context)->value += 2; return true; }
bool pack(void* context) noexcept { static_cast<state*>(context)->value += 3; return true; }
bool execute(void* context) noexcept {
    auto* current = static_cast<state*>(context);
    for (int i = 0; i < 32; ++i) current->value += static_cast<unsigned>(i);
    return true;
}
bool baseline(void* context) noexcept {
    auto* current = static_cast<state*>(context);
    for (int i = 0; i < 32; ++i) current->value += static_cast<unsigned>(i);
    return true;
}
bool referee(void*) noexcept { return true; }
}  // namespace

int main() {
    state context{};
    const auto result = cb::benchmark_cpu_backend_complete_cost_v1({compile,
        prepare, pack, execute, baseline, referee, &context, 4, 1000, 4096, 2048});
    assert(result.compile_nanoseconds > 0);
    assert(result.preparation_nanoseconds > 0);
    assert(result.packing_nanoseconds > 0);
    assert(result.execution_nanoseconds > 0);
    assert(result.warm_reuse_nanoseconds > 0);
    assert(result.plain_cpp_nanoseconds > 0);
    assert(result.generated_source_bytes == 4096 && result.object_bytes == 2048);
    assert(result.semantic_referee_passed);
    assert(result.evaluation == cb::cpu_backend_evaluation_v1::evaluated_promoted
        || result.evaluation
            == cb::cpu_backend_evaluation_v1::evaluated_not_promoted);
}
