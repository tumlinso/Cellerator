#include <Cellerator/compiler/ast/freeze_ast_node_ownership_and_lifetime_v1.hh>

#include <limits>
#include <stdexcept>
#include <utility>

namespace Cellerator::compiler::ast {
namespace {

template <typename Record>
void reserve_for_append(std::vector<Record>& records, std::size_t& allocations) {
    if (records.size() != records.capacity()) {
        return;
    }
    const auto next = records.capacity() == 0 ? std::size_t{64} : records.capacity() * 2;
    records.reserve(next);
    ++allocations;
}

} // namespace

struct ast_snapshot_v1::storage_v1 {
    ast_arena_id_v1 arena_id = 0;
    std::vector<ast_node_record_v1> nodes;
    std::vector<ast_region_record_v1> regions;
    std::size_t allocation_count = 0;
};

struct ast_arena_v1::builder_storage_v1 {
    ast_arena_id_v1 arena_id = 0;
    bool sealed = false;
    std::vector<ast_node_record_v1> nodes;
    std::vector<ast_region_record_v1> regions;
    std::size_t allocation_count = 0;
};

ast_snapshot_v1::ast_snapshot_v1(std::shared_ptr<const storage_v1> storage) noexcept
    : storage_(std::move(storage)) {}

ast_arena_id_v1 ast_snapshot_v1::arena_id() const noexcept {
    return storage_ ? storage_->arena_id : 0;
}

std::size_t ast_snapshot_v1::node_count() const noexcept {
    return storage_ ? storage_->nodes.size() : 0;
}

std::size_t ast_snapshot_v1::region_count() const noexcept {
    return storage_ ? storage_->regions.size() : 0;
}

const ast_node_record_v1* ast_snapshot_v1::node(ast_node_handle_v1 handle) const noexcept {
    if (!storage_ || handle.arena != storage_->arena_id || handle.slot >= storage_->nodes.size()) {
        return nullptr;
    }
    return &storage_->nodes[handle.slot];
}

const ast_region_record_v1* ast_snapshot_v1::region(ast_region_handle_v1 handle) const noexcept {
    if (!storage_ || handle.arena != storage_->arena_id || handle.slot >= storage_->regions.size()) {
        return nullptr;
    }
    return &storage_->regions[handle.slot];
}

ast_arena_metrics_v1 ast_snapshot_v1::metrics() const noexcept {
    if (!storage_) {
        return {};
    }
    return {storage_->nodes.size(), storage_->regions.size(),
            storage_->nodes.size() * sizeof(ast_node_record_v1),
            storage_->regions.size() * sizeof(ast_region_record_v1),
            storage_->allocation_count,
            storage_->nodes.capacity() * sizeof(ast_node_record_v1) +
                storage_->regions.capacity() * sizeof(ast_region_record_v1)};
}

bool ast_snapshot_v1::shares_storage_with(const ast_snapshot_v1& other) const noexcept {
    return storage_ && storage_ == other.storage_;
}

ast_arena_v1::ast_arena_v1(ast_arena_id_v1 arena_id)
    : storage_(std::make_unique<builder_storage_v1>()) {
    if (arena_id == 0) {
        throw std::invalid_argument("AST arena identity must be nonzero");
    }
    storage_->arena_id = arena_id;
}

ast_arena_v1::ast_arena_v1(ast_arena_v1&&) noexcept = default;
ast_arena_v1& ast_arena_v1::operator=(ast_arena_v1&&) noexcept = default;
ast_arena_v1::~ast_arena_v1() = default;

ast_arena_id_v1 ast_arena_v1::arena_id() const noexcept {
    return storage_ ? storage_->arena_id : 0;
}

bool ast_arena_v1::sealed() const noexcept {
    return !storage_ || storage_->sealed;
}

std::optional<ast_region_handle_v1>
ast_arena_v1::append_region(ast_region_handle_v1 parent, ast_node_handle_v1 lexical_owner) {
    if (!storage_ || storage_->sealed ||
        storage_->regions.size() >= invalid_ast_slot_v1) {
        return std::nullopt;
    }
    if (parent.valid() &&
        (parent.arena != storage_->arena_id || parent.slot >= storage_->regions.size())) {
        return std::nullopt;
    }
    if (lexical_owner.valid() &&
        (lexical_owner.arena != storage_->arena_id || lexical_owner.slot >= storage_->nodes.size())) {
        return std::nullopt;
    }
    reserve_for_append(storage_->regions, storage_->allocation_count);
    const ast_region_handle_v1 handle{storage_->arena_id,
                                      static_cast<std::uint32_t>(storage_->regions.size())};
    storage_->regions.push_back({handle, parent, lexical_owner});
    return handle;
}

std::optional<ast_node_handle_v1>
ast_arena_v1::append_node(ast_node_class_v1 node_class, ast_node_handle_v1 parent,
                          ast_region_handle_v1 region, source_identity_v1 source_identity,
                          std::uint16_t flags) {
    if (!storage_ || storage_->sealed || node_class == ast_node_class_v1::unknown ||
        !region.valid() || region.arena != storage_->arena_id ||
        region.slot >= storage_->regions.size() || storage_->nodes.size() >= invalid_ast_slot_v1) {
        return std::nullopt;
    }
    if (parent.valid() &&
        (parent.arena != storage_->arena_id || parent.slot >= storage_->nodes.size())) {
        return std::nullopt;
    }
    reserve_for_append(storage_->nodes, storage_->allocation_count);
    const ast_node_handle_v1 handle{storage_->arena_id,
                                    static_cast<std::uint32_t>(storage_->nodes.size())};
    storage_->nodes.push_back({handle, parent, region, source_identity, node_class, flags});
    return handle;
}

ast_arena_metrics_v1 ast_arena_v1::metrics() const noexcept {
    if (!storage_) {
        return {};
    }
    return {storage_->nodes.size(), storage_->regions.size(),
            storage_->nodes.size() * sizeof(ast_node_record_v1),
            storage_->regions.size() * sizeof(ast_region_record_v1),
            storage_->allocation_count,
            storage_->nodes.capacity() * sizeof(ast_node_record_v1) +
                storage_->regions.capacity() * sizeof(ast_region_record_v1)};
}

ast_snapshot_v1 ast_arena_v1::freeze() && {
    if (!storage_ || storage_->sealed) {
        return {};
    }
    storage_->sealed = true;
    auto frozen = std::make_shared<ast_snapshot_v1::storage_v1>();
    frozen->arena_id = storage_->arena_id;
    frozen->nodes = std::move(storage_->nodes);
    frozen->regions = std::move(storage_->regions);
    frozen->allocation_count = storage_->allocation_count + 1;
    storage_.reset();
    return ast_snapshot_v1{std::move(frozen)};
}

} // namespace Cellerator::compiler::ast
