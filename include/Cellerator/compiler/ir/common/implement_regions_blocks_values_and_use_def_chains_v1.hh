#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compiler::ir {

template<class Tag> struct graph_handle {
    std::uint32_t slot{};
    std::uint32_t generation{};
};
struct region_tag; struct block_tag; struct value_tag; struct operation_tag;
using region_handle = graph_handle<region_tag>;
using block_handle = graph_handle<block_tag>;
using value_handle = graph_handle<value_tag>;
using operation_handle = graph_handle<operation_tag>;

struct value_use { operation_handle user{}; std::uint32_t operand{}; };
struct value_record {
    std::string type;
    operation_handle definition{};
    std::vector<value_use> uses;
};
struct block_record {
    region_handle parent{};
    std::vector<operation_handle> operations;
    std::vector<block_handle> successors;
};
struct region_record { operation_handle parent{}; std::vector<block_handle> blocks; };
struct operation_record {
    block_handle parent{};
    std::vector<value_handle> operands;
    std::vector<value_handle> results;
    std::vector<region_handle> regions;
};

class ir_graph {
public:
    region_handle add_region(operation_handle parent = {});
    block_handle add_block(region_handle parent);
    operation_handle add_operation(block_handle parent,
        const std::vector<value_handle> &operands = {});
    value_handle add_value(operation_handle definition, std::string type);
    bool add_control_edge(block_handle from, block_handle to);
    bool replace_all_uses(value_handle from, value_handle to);
    const region_record *region(region_handle handle) const noexcept;
    const block_record *block(block_handle handle) const noexcept;
    const operation_record *operation(operation_handle handle) const noexcept;
    const value_record *value(value_handle handle) const noexcept;
private:
    std::uint32_t generation_{1u};
    std::vector<region_record> regions_;
    std::vector<block_record> blocks_;
    std::vector<operation_record> operations_;
    std::vector<value_record> values_;
};

} // namespace cellerator::compiler::ir
