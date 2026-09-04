#include <Cellerator/compiler/ir/common/implement_regions_blocks_values_and_use_def_chains_v1.hh>

#include <algorithm>

namespace cellerator::compiler::ir {
namespace {
template<class Record, class Handle>
const Record *resolve(const std::vector<Record> &records, Handle handle,
    std::uint32_t generation) noexcept {
    return handle.generation == generation && handle.slot < records.size()
        ? &records[handle.slot] : nullptr;
}
} // namespace

region_handle ir_graph::add_region(operation_handle parent) {
    const region_handle handle{static_cast<std::uint32_t>(regions_.size()), generation_};
    regions_.push_back({parent, {}});
    if (auto *owner = const_cast<operation_record *>(operation(parent)))
        owner->regions.push_back(handle);
    return handle;
}

block_handle ir_graph::add_block(region_handle parent) {
    const block_handle handle{static_cast<std::uint32_t>(blocks_.size()), generation_};
    blocks_.push_back({parent, {}, {}});
    if (auto *owner = const_cast<region_record *>(region(parent)))
        owner->blocks.push_back(handle);
    return handle;
}

operation_handle ir_graph::add_operation(block_handle parent,
    const std::vector<value_handle> &operands) {
    const operation_handle handle{
        static_cast<std::uint32_t>(operations_.size()), generation_};
    operations_.push_back({parent, operands, {}, {}});
    if (auto *owner = const_cast<block_record *>(block(parent)))
        owner->operations.push_back(handle);
    for (std::size_t index = 0; index < operands.size(); ++index) {
        if (auto *operand = const_cast<value_record *>(value(operands[index])))
            operand->uses.push_back({handle, static_cast<std::uint32_t>(index)});
    }
    return handle;
}

value_handle ir_graph::add_value(operation_handle definition, std::string type) {
    const value_handle handle{static_cast<std::uint32_t>(values_.size()), generation_};
    values_.push_back({std::move(type), definition, {}});
    if (auto *owner = const_cast<operation_record *>(operation(definition)))
        owner->results.push_back(handle);
    return handle;
}

bool ir_graph::add_control_edge(block_handle from, block_handle to) {
    auto *source = const_cast<block_record *>(block(from));
    if (!source || !block(to))
        return false;
    if (std::find_if(source->successors.begin(), source->successors.end(),
            [to](block_handle value) { return value.slot == to.slot && value.generation == to.generation; })
        == source->successors.end())
        source->successors.push_back(to);
    return true;
}

bool ir_graph::replace_all_uses(value_handle from, value_handle to) {
    auto *source = const_cast<value_record *>(value(from));
    auto *destination = const_cast<value_record *>(value(to));
    if (!source || !destination || source->type != destination->type)
        return false;
    for (const auto use : source->uses) {
        auto *user = const_cast<operation_record *>(operation(use.user));
        if (!user || use.operand >= user->operands.size())
            return false;
        user->operands[use.operand] = to;
        destination->uses.push_back(use);
    }
    source->uses.clear();
    return true;
}

const region_record *ir_graph::region(region_handle handle) const noexcept {
    return resolve(regions_, handle, generation_);
}
const block_record *ir_graph::block(block_handle handle) const noexcept {
    return resolve(blocks_, handle, generation_);
}
const operation_record *ir_graph::operation(operation_handle handle) const noexcept {
    return resolve(operations_, handle, generation_);
}
const value_record *ir_graph::value(value_handle handle) const noexcept {
    return resolve(values_, handle, generation_);
}

} // namespace cellerator::compiler::ir
