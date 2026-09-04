#include <Cellerator/compiler/ir/common/implement_arena_and_ownership_model_v1.hh>

#include <cassert>
#include <chrono>

using namespace cellerator::compiler::ir;

int main() {
    ir_context context;
    arena_builder builder(context);
    constexpr std::size_t count = 100000u;
    builder.reserve(count);
    node_handle last{};
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < count; ++index)
        last = builder.append({7u, {}, {}, index});
    const auto snapshot = builder.freeze();
    std::uint64_t total = 0u;
    for (const auto &node : snapshot.nodes())
        total += node.payload;
    assert(total == (count - 1u) * count / 2u);
    assert(snapshot.resolve(last)->payload == count - 1u);
    assert(snapshot.metrics().bytes_per_node == sizeof(arena_node));
    assert(snapshot.metrics().allocation_events <= 2u);

    arena_builder edit(context, snapshot);
    const auto edit_handle = edit.append({9u, {}, {}, 11u});
    const auto edited = edit.freeze();
    assert(snapshot.nodes().size == count);
    assert(edited.nodes().size == count + 1u);
    assert(edited.resolve(edit_handle)->opcode == 9u);
    assert(edit.metrics().copied_nodes == count);
    assert(std::chrono::steady_clock::now() >= start);
}
