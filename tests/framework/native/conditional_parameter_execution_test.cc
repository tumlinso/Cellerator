#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/execution/conditional/parameter_dispatch.hh>

#include <cassert>

namespace c = cellerator::execution::conditional;

c::ParameterBuffers weights(float value) {
    auto storage = std::make_shared<const std::vector<float>>(std::vector<float>{value});
    c::ParameterBuffers buffers;
    buffers.parameters = {storage, storage->data(), sizeof(float), "float32", "host"};
    return buffers;
}

int main() {
    c::ParameterStore store;
    store.register_initial({1, 1}, weights(2));
    store.register_initial({2, 3}, weights(-1));
    const c::Origin first{0, 100, 7}, second{0, 200, 7};
    c::DispatchPlan plan({{{1,1}, first, 0.25}, {{2,3}, first, 0.75}, {{1,1}, second, 1}});
    bool retired_while_executing = false;
    const auto result = c::execute_parameterized<float,float,float,float>(plan, store, 1, 1,
        [&](const c::Origin& origin, float* row, std::size_t) { row[0] = origin == first ? 3 : 4; },
        [&](const c::ParameterRevision& revision, const c::InputBatch<float>& input, c::OutputBatch<float>& output) {
            if (revision.key == c::ExpertKey{1,1}) {
                // All expert versions were pinned before callbacks. A newer
                // revision and retirement cannot alter the remaining callback.
                store.publish({2,3}, {2,4}, weights(100));
                assert(store.retire({2,3}));
                retired_while_executing = true;
            } else {
                assert(retired_while_executing && revision.key == (c::ExpertKey{2,3}));
            }
            const float weight = *static_cast<const float*>(revision.buffers.parameters.data);
            for (std::size_t row = 0; row < input.rows; ++row) output.at(row,0) = weight * input.at(row,0);
        });
    assert((result == std::vector<float>{-0.75f, 8.0f}));
    assert(store.current(2) == (c::ExpertKey{2,4}));

    // A retired required revision fails before the input provider or any
    // expert callback, instead of silently substituting its current revision.
    std::size_t reads = 0, calls = 0;
    bool rejected = false;
    try {
        c::execute_parameterized<float>(plan, store, 1, 1,
            [&](const c::Origin&, float*, std::size_t) { ++reads; },
            [&](const c::ParameterRevision&, const c::InputBatch<float>&, c::OutputBatch<float>&) { ++calls; });
    } catch (const std::out_of_range&) { rejected = true; }
    assert(rejected && reads == 0 && calls == 0);

    // Selecting the new revision explicitly changes the computation.
    c::DispatchPlan next({{{2,4}, first, 1}});
    const auto changed = c::execute_parameterized<float>(next, store, 1, 1,
        [](const c::Origin&, float* row, std::size_t) { row[0] = 3; },
        [](const c::ParameterRevision& revision, const c::InputBatch<float>& input, c::OutputBatch<float>& output) {
            output.at(0,0) = *static_cast<const float*>(revision.buffers.parameters.data) * input.at(0,0);
        });
    assert(changed[0] == 300);
}
