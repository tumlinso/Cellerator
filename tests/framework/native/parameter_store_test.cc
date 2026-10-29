#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/execution/conditional/parameter_store.hh>
#include <cassert>
#include <future>
#include <iostream>
#include <vector>
using namespace cellerator::execution::conditional;

ParameterBuffer buffer(std::initializer_list<float> values) {
    auto owner = std::make_shared<std::vector<float>>(values);
    return {owner, owner->data(), owner->size() * sizeof(float), "float32", "host"};
}
float first(const ParameterBuffer& value) { return static_cast<const float*>(value.data)[0]; }
int main() {
    ParameterStore store;
    ParameterBuffers initial;
    initial.parameters = buffer({3, 4});
    initial.master_parameters = buffer({3, 4});
    initial.optimizer_state = buffer({0, 0});
    store.register_initial({9, 0}, initial);
    auto execution = store.acquire({9, 0});
    assert(store.leases({9, 0}) == 1);
    auto gradient = buffer({2, 2});
    auto next = store.update({9, 0}, 1, gradient,
        [](const ParameterRevision& old, const ParameterBuffer& grad) {
            ParameterBuffers result;
            result.parameters = buffer({first(old.buffers.parameters) - 0.5f * first(grad), 3});
            result.master_parameters = result.parameters;
            result.gradients = grad;
            result.optimizer_state = buffer({1, 1});
            return result;
        });
    assert(next == (ExpertKey{9, 1}));
    assert(first(execution.buffers().parameters) == 3);
    assert(first(store.acquire(next).buffers().parameters) == 2);
    assert(first(store.acquire(next).buffers().gradients) == 2);
    assert(store.retire({9, 0}));
    assert(first(execution.buffers().parameters) == 3); // old asynchronous reader survives
    bool missing = false;
    try { (void)store.acquire({9, 0}); } catch (const std::out_of_range&) { missing = true; }
    assert(missing);
    bool stale = false;
    try { store.publish({9, 0}, {9, 2}, initial); }
    catch (const StaleParameterRevision&) { stale = true; }
    assert(stale && store.current(9) == next);
    bool current_retirement = false;
    try { store.retire(next); } catch (const std::logic_error&) { current_retirement = true; }
    assert(current_retirement);
    auto snapshot = store.snapshot({next, next});
    assert(snapshot.size() == 2 && store.leases(next) == 2);
    bool bad_buffer = false;
    try { ParameterBuffer{{}, nullptr, 4, "float32", "host"}.validate(); }
    catch (const std::invalid_argument&) { bad_buffer = true; }
    assert(bad_buffer);
    // Race two optimizers from the same base; only one publication is accepted.
    std::promise<void> started, proceed;
    auto proceed_future = proceed.get_future();
    auto racing = std::async(std::launch::async, [&] {
        try {
            store.update(next, 2, gradient, [&](const auto&, const auto&) {
                started.set_value(); proceed_future.wait(); return initial;
            });
            return false;
        } catch (const StaleParameterRevision&) { return true; }
    });
    started.get_future().wait();
    store.publish(next, {9, 3}, initial);
    proceed.set_value();
    assert(racing.get());
    assert(store.current(9) == (ExpertKey{9, 3}));
    execution.release();
    bool released = false;
    try { execution.revision(); } catch (const std::logic_error&) { released = true; }
    assert(released);
    std::cout << "parameter store lifecycle and concurrent update checks passed\n";
}
