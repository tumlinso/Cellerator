#include <Cellerator/compiler/ast/freeze_ast_node_ownership_and_lifetime_v1.hh>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <sys/resource.h>
#include <type_traits>
#include <vector>

using namespace Cellerator::compiler::ast;

namespace {

std::size_t peak_rss_bytes() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        throw std::runtime_error("getrusage failed");
    }
    return static_cast<std::size_t>(usage.ru_maxrss) * 1024U;
}

} // namespace

int main() {
    try {
        static_assert(std::is_trivially_copyable_v<ast_node_record_v1>);
        static_assert(std::is_trivially_copyable_v<ast_region_record_v1>);

        constexpr std::size_t node_count = 250000;
        ast_arena_v1 arena{0xC020001U};
        const auto root_region = arena.append_region();
        if (!root_region) {
            throw std::runtime_error("root region was not created");
        }

        std::vector<ast_node_handle_v1> sampled_handles;
        sampled_handles.reserve(256);
        ast_node_handle_v1 parent{};
        const auto started = std::chrono::steady_clock::now();
        for (std::size_t index = 0; index < node_count; ++index) {
            const auto handle = arena.append_node(
                index == 0 ? ast_node_class_v1::translation_unit
                           : ast_node_class_v1::declaration,
                parent, *root_region, static_cast<source_identity_v1>(index + 1));
            if (!handle) {
                throw std::runtime_error("large AST append failed");
            }
            if ((index % 1024U) == 0) {
                sampled_handles.push_back(*handle);
            }
            parent = *handle;
        }
        const auto elapsed = std::chrono::steady_clock::now() - started;
        const auto construction_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();

        const auto before = arena.metrics();
        if (before.node_count != node_count || before.region_count != 1 ||
            before.node_bytes != node_count * sizeof(ast_node_record_v1) ||
            before.allocation_count > 16 || before.reserved_bytes < before.node_bytes) {
            throw std::runtime_error("AST construction metrics are inconsistent");
        }

        ast_arena_v1 foreign{0xC020002U};
        const auto foreign_region = foreign.append_region();
        if (!foreign_region || arena.append_node(ast_node_class_v1::declaration, {},
                                                  *foreign_region, 7)) {
            throw std::runtime_error("cross-arena region was accepted");
        }

        auto snapshot = std::move(arena).freeze();
        auto shared_snapshot = snapshot;
        if (!snapshot.shares_storage_with(shared_snapshot) || snapshot.node_count() != node_count) {
            throw std::runtime_error("snapshot storage was copied instead of shared");
        }
        for (const auto handle : sampled_handles) {
            const auto* node = snapshot.node(handle);
            if (node == nullptr || !(node->handle == handle) ||
                node->source_identity != static_cast<source_identity_v1>(handle.slot + 1)) {
                throw std::runtime_error("stable handle did not survive freeze");
            }
        }
        if (snapshot.node({0xC020002U, 0}) != nullptr ||
            snapshot.region({snapshot.arena_id(), invalid_ast_slot_v1}) != nullptr) {
            throw std::runtime_error("invalid or foreign handle was resolved");
        }

        const auto frozen = snapshot.metrics();
        const auto per_second = construction_ns == 0
                                    ? 0
                                    : (node_count * 1000000000ULL) /
                                          static_cast<std::uint64_t>(construction_ns);
        std::cout << "nodes=" << frozen.node_count
                  << " node_bytes=" << sizeof(ast_node_record_v1)
                  << " allocations=" << frozen.allocation_count
                  << " construction_nodes_per_second=" << per_second
                  << " peak_rss_bytes=" << peak_rss_bytes() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
