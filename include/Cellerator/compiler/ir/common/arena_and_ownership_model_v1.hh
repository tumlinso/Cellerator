#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace cellerator::compiler::ir {

struct node_handle {
    std::uint32_t slot{};
    std::uint32_t generation{};
};

struct arena_node {
    std::uint32_t opcode{};
    node_handle parent{};
    node_handle region{};
    std::uint64_t payload{};
};

struct node_view {
    const arena_node *data{};
    std::size_t size{};
    const arena_node *begin() const noexcept { return data; }
    const arena_node *end() const noexcept { return data + size; }
};

struct arena_metrics {
    std::size_t bytes_per_node{};
    std::size_t storage_bytes{};
    std::size_t allocation_events{};
    std::size_t copied_nodes{};
};

class ir_context {
public:
    std::uint32_t next_generation() noexcept;
private:
    std::uint32_t generation_{1u};
};

class immutable_arena {
public:
    immutable_arena() = default;
    node_view nodes() const noexcept;
    const arena_node *resolve(node_handle handle) const noexcept;
    arena_metrics metrics() const noexcept;
private:
    friend class arena_builder;
    immutable_arena(std::shared_ptr<const std::vector<arena_node>> storage,
        std::uint32_t generation, arena_metrics metrics) noexcept;
    std::shared_ptr<const std::vector<arena_node>> storage_{};
    std::uint32_t generation_{};
    arena_metrics metrics_{};
};

class arena_builder {
public:
    explicit arena_builder(ir_context &context);
    arena_builder(ir_context &context, const immutable_arena &base);
    void reserve(std::size_t nodes);
    node_handle append(arena_node node);
    bool replace(node_handle handle, arena_node node);
    immutable_arena freeze() const;
    arena_metrics metrics() const noexcept;
private:
    void make_writable();
    ir_context *context_{};
    std::shared_ptr<std::vector<arena_node>> storage_{};
    std::uint32_t generation_{};
    arena_metrics metrics_{};
    bool shared_base_{};
};

} // namespace cellerator::compiler::ir
