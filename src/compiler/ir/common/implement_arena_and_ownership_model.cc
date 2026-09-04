#include <Cellerator/compiler/ir/common/implement_arena_and_ownership_model_v1.hh>

#include <limits>

namespace cellerator::compiler::ir {

std::uint32_t ir_context::next_generation() noexcept {
    if (generation_ == std::numeric_limits<std::uint32_t>::max())
        generation_ = 1u;
    return generation_++;
}

immutable_arena::immutable_arena(
    std::shared_ptr<const std::vector<arena_node>> storage,
    std::uint32_t generation, arena_metrics metrics) noexcept
    : storage_(std::move(storage)), generation_(generation), metrics_(metrics) {}

node_view immutable_arena::nodes() const noexcept {
    return storage_ ? node_view{storage_->data(), storage_->size()} : node_view{};
}

const arena_node *immutable_arena::resolve(node_handle handle) const noexcept {
    if (!storage_ || handle.generation != generation_ || handle.slot >= storage_->size())
        return nullptr;
    return &(*storage_)[handle.slot];
}

arena_metrics immutable_arena::metrics() const noexcept { return metrics_; }

arena_builder::arena_builder(ir_context &context)
    : context_(&context), storage_(std::make_shared<std::vector<arena_node>>()),
      generation_(context.next_generation()) {
    metrics_.allocation_events = 1u;
    metrics_.bytes_per_node = sizeof(arena_node);
}

arena_builder::arena_builder(ir_context &context, const immutable_arena &base)
    : context_(&context), generation_(context.next_generation()), metrics_(base.metrics_),
      shared_base_(true) {
    storage_ = std::const_pointer_cast<std::vector<arena_node>>(base.storage_);
}

void arena_builder::make_writable() {
    if (!storage_)
        storage_ = std::make_shared<std::vector<arena_node>>();
    else if (shared_base_ || !storage_.unique()) {
        metrics_.copied_nodes += storage_->size();
        storage_ = std::make_shared<std::vector<arena_node>>(*storage_);
        ++metrics_.allocation_events;
    }
    shared_base_ = false;
}

void arena_builder::reserve(std::size_t nodes) {
    make_writable();
    if (nodes > storage_->capacity()) {
        storage_->reserve(nodes);
        ++metrics_.allocation_events;
    }
}

node_handle arena_builder::append(arena_node node) {
    make_writable();
    const auto slot = static_cast<std::uint32_t>(storage_->size());
    storage_->push_back(node);
    metrics_.storage_bytes = storage_->capacity() * sizeof(arena_node);
    return {slot, generation_};
}

bool arena_builder::replace(node_handle handle, arena_node node) {
    if (handle.generation != generation_ || !storage_ || handle.slot >= storage_->size())
        return false;
    make_writable();
    (*storage_)[handle.slot] = node;
    return true;
}

immutable_arena arena_builder::freeze() const {
    auto metrics = metrics_;
    metrics.storage_bytes = storage_ ? storage_->capacity() * sizeof(arena_node) : 0u;
    return immutable_arena(storage_, generation_, metrics);
}

arena_metrics arena_builder::metrics() const noexcept { return metrics_; }

} // namespace cellerator::compiler::ir
