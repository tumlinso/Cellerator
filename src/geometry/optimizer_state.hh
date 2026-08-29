#pragma once

#include "Cellerator/geometry/optimizer.hh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace cellpack::detail {

struct alignas(16) optimizer_block_desc {
    u32 stable_key = invalid_id;
    u32 generation = 0u;
    u32 member_count = 0u;
    u32 active = 0u;
    mutable u64 union_count = 0u;
    mutable u32 union_valid = 0u;
    u32 reserved = 0u;
};

static_assert(sizeof(optimizer_block_desc) == 32u,
    "optimizer block descriptors must remain compact and aligned");

// Slots never own storage. All mutation-time state is prepared in coherent
// fixed-size tables; merge, move, and swap never grow or reallocate them.
class optimizer_state {
public:
    validation_result initialize(
        u32 row_count,
        const sampled_feature_support_view &support,
        u32 maximum_block_width,
        u32 row_group_width) {
        if (support.feature_count == 0u || maximum_block_width == 0u || row_group_width == 0u) {
            return validation_error(validation_code::invalid_plan_geometry, invalid_id,
                "optimizer dimensions and configured widths must be nonzero");
        }
        if (static_cast<std::size_t>(support.feature_count) >
            std::numeric_limits<std::size_t>::max() / maximum_block_width
            || (support.words_per_feature != 0u
                && static_cast<std::size_t>(support.feature_count) >
                    std::numeric_limits<std::size_t>::max() / support.words_per_feature)) {
            return validation_error(validation_code::integer_overflow, invalid_id,
                "optimizer prepared table size overflows");
        }
        support_ = support;
        row_count_ = row_count;
        maximum_block_width_ = maximum_block_width;
        row_group_width_ = row_group_width;
        blocks_.assign(support.feature_count, optimizer_block_desc{});
        members_.assign(static_cast<std::size_t>(support.feature_count) * maximum_block_width, invalid_id);
        union_words_.assign(
            static_cast<std::size_t>(support.feature_count) * support.words_per_feature, 0u);
        feature_to_slot_.resize(support.feature_count);
        merge_scratch_.resize(maximum_block_width);
        for (u32 feature = 0u; feature < support.feature_count; ++feature) {
            blocks_[feature].active = 1u;
            blocks_[feature].stable_key = feature;
            blocks_[feature].member_count = 1u;
            member_row(feature)[0] = feature;
            feature_to_slot_[feature] = feature;
        }
        reserve_materialization();
        dirty_ = true;
        return validate();
    }

    u32 row_count() const noexcept { return row_count_; }
    u32 feature_count() const noexcept { return support_.feature_count; }
    u32 maximum_block_width() const noexcept { return maximum_block_width_; }
    u32 row_group_width() const noexcept { return row_group_width_; }
    bool dirty() const noexcept { return dirty_; }
    u32 block_slot_for_feature(u32 feature) const noexcept {
        return feature < feature_to_slot_.size() ? feature_to_slot_[feature] : invalid_id;
    }
    u32 block_width(u32 slot) const noexcept {
        return block_active(slot) ? blocks_[slot].member_count : 0u;
    }
    u32 block_stable_key(u32 slot) const noexcept {
        return block_active(slot) ? blocks_[slot].stable_key : invalid_id;
    }
    u32 block_generation(u32 slot) const noexcept {
        return slot < blocks_.size() ? blocks_[slot].generation : 0u;
    }
    bool block_active(u32 slot) const noexcept {
        return slot < blocks_.size() && blocks_[slot].active != 0u;
    }
    const u32 *block_members(u32 slot) const noexcept {
        return slot < blocks_.size() ? member_row(slot) : nullptr;
    }
    u32 active_block_count() const noexcept {
        u32 result = 0u;
        for (const optimizer_block_desc &block : blocks_) result += block.active != 0u ? 1u : 0u;
        return result;
    }

    validation_result merge_blocks(u32 lhs_slot, u32 rhs_slot) {
        if (!legal_distinct_blocks(lhs_slot, rhs_slot)) {
            return validation_error(validation_code::invalid_plan_geometry, invalid_id,
                "merge requires two distinct active blocks");
        }
        if (block_width(lhs_slot) + block_width(rhs_slot) > maximum_block_width_) {
            return validation_error(validation_code::invalid_plan_geometry, invalid_id,
                "merge exceeds maximum block width");
        }
        if (blocks_[rhs_slot].stable_key < blocks_[lhs_slot].stable_key) std::swap(lhs_slot, rhs_slot);
        optimizer_block_desc &lhs = blocks_[lhs_slot];
        optimizer_block_desc &rhs = blocks_[rhs_slot];
        std::merge(member_row(lhs_slot), member_row(lhs_slot) + lhs.member_count,
            member_row(rhs_slot), member_row(rhs_slot) + rhs.member_count,
            merge_scratch_.begin());
        lhs.member_count += rhs.member_count;
        std::copy_n(merge_scratch_.data(), lhs.member_count, member_row(lhs_slot));
        clear_member_tail(lhs_slot, lhs.member_count);
        lhs.stable_key = member_row(lhs_slot)[0];
        ++lhs.generation;
        invalidate_union(lhs_slot);
        for (u32 index = 0u; index < rhs.member_count; ++index) {
            feature_to_slot_[member_row(rhs_slot)[index]] = lhs_slot;
        }
        rhs.active = 0u;
        rhs.member_count = 0u;
        rhs.stable_key = invalid_id;
        ++rhs.generation;
        invalidate_union(rhs_slot);
        clear_member_tail(rhs_slot, 0u);
        dirty_ = true;
        return validate_touched(lhs_slot, rhs_slot);
    }

    validation_result move_feature(u32 feature, u32 destination_slot) {
        if (feature >= feature_count() || !block_active(destination_slot)) {
            return validation_error(validation_code::invalid_plan_geometry, feature,
                "move feature or destination is invalid");
        }
        const u32 source_slot = feature_to_slot_[feature];
        if (source_slot == destination_slot) {
            return validation_error(validation_code::invalid_plan_geometry, feature,
                "move destination equals source block");
        }
        if (block_width(destination_slot) >= maximum_block_width_) {
            return validation_error(validation_code::invalid_plan_geometry, destination_slot,
                "move destination is at maximum width");
        }
        optimizer_block_desc &source = blocks_[source_slot];
        optimizer_block_desc &destination = blocks_[destination_slot];
        erase_member(source_slot, feature);
        insert_member(destination_slot, feature);
        feature_to_slot_[feature] = destination_slot;
        ++source.generation;
        ++destination.generation;
        invalidate_union(source_slot);
        invalidate_union(destination_slot);
        if (source.member_count == 0u) {
            source.active = 0u;
            source.stable_key = invalid_id;
        } else {
            source.stable_key = member_row(source_slot)[0];
        }
        destination.stable_key = member_row(destination_slot)[0];
        dirty_ = true;
        return validate_touched(source_slot, destination_slot);
    }

    validation_result swap_features(u32 feature_a, u32 feature_b) {
        if (feature_a >= feature_count() || feature_b >= feature_count() || feature_a == feature_b) {
            return validation_error(validation_code::invalid_plan_geometry, invalid_id,
                "swap features are invalid");
        }
        const u32 slot_a = feature_to_slot_[feature_a];
        const u32 slot_b = feature_to_slot_[feature_b];
        if (slot_a == slot_b) {
            return validation_error(validation_code::invalid_plan_geometry, feature_a,
                "swap features already share a block");
        }
        replace_member(slot_a, feature_a, feature_b);
        replace_member(slot_b, feature_b, feature_a);
        feature_to_slot_[feature_a] = slot_b;
        feature_to_slot_[feature_b] = slot_a;
        optimizer_block_desc &block_a = blocks_[slot_a];
        optimizer_block_desc &block_b = blocks_[slot_b];
        block_a.stable_key = member_row(slot_a)[0];
        block_b.stable_key = member_row(slot_b)[0];
        ++block_a.generation;
        ++block_b.generation;
        invalidate_union(slot_a);
        invalidate_union(slot_b);
        dirty_ = true;
        return validate_touched(slot_a, slot_b);
    }

    std::int64_t merge_proxy_gain(u32 lhs_slot, u32 rhs_slot) const {
        if (!legal_distinct_blocks(lhs_slot, rhs_slot)) return std::numeric_limits<std::int64_t>::min();
        ensure_union(lhs_slot);
        ensure_union(rhs_slot);
        u64 intersection = 0u;
        for (u32 word = 0u; word < support_.words_per_feature; ++word) {
            intersection += popcount(mask_tail(word,
                union_word(lhs_slot, word) & union_word(rhs_slot, word)));
        }
        return static_cast<std::int64_t>(intersection);
    }

    std::int64_t move_proxy_gain(u32 feature, u32 destination_slot) const {
        if (feature >= feature_count() || !block_active(destination_slot)) return std::numeric_limits<std::int64_t>::min();
        const u32 source_slot = feature_to_slot_[feature];
        if (source_slot == destination_slot || block_width(destination_slot) >= maximum_block_width_) {
            return std::numeric_limits<std::int64_t>::min();
        }
        ensure_union(source_slot);
        ensure_union(destination_slot);
        const u64 before = blocks_[source_slot].union_count + blocks_[destination_slot].union_count;
        const u64 after = union_count_modified(source_slot, feature, invalid_id)
            + union_count_modified(destination_slot, invalid_id, feature);
        return signed_difference(before, after);
    }

    std::int64_t swap_proxy_gain(u32 feature_a, u32 feature_b) const {
        if (feature_a >= feature_count() || feature_b >= feature_count() || feature_a == feature_b) {
            return std::numeric_limits<std::int64_t>::min();
        }
        const u32 slot_a = feature_to_slot_[feature_a], slot_b = feature_to_slot_[feature_b];
        if (slot_a == slot_b) return std::numeric_limits<std::int64_t>::min();
        ensure_union(slot_a);
        ensure_union(slot_b);
        const u64 before = blocks_[slot_a].union_count + blocks_[slot_b].union_count;
        const u64 after = union_count_modified(slot_a, feature_a, feature_b)
            + union_count_modified(slot_b, feature_b, feature_a);
        return signed_difference(before, after);
    }

    validation_result materialize_execution_geometry() {
        const validation_result state_status = validate();
        if (!state_status) return state_status;
        active_slots_.clear();
        for (u32 slot = 0u; slot < blocks_.size(); ++slot) if (block_active(slot)) active_slots_.push_back(slot);
        std::sort(active_slots_.begin(), active_slots_.end(), [&](u32 lhs, u32 rhs) {
            return blocks_[lhs].stable_key < blocks_[rhs].stable_key;
        });
        feature_permutation_.clear();
        feature_block_offsets_.clear();
        feature_block_offsets_.push_back(0u);
        for (u32 slot : active_slots_) {
            feature_permutation_.insert(feature_permutation_.end(),
                member_row(slot), member_row(slot) + block_width(slot));
            feature_block_offsets_.push_back(static_cast<u32>(feature_permutation_.size()));
        }
        std::fill(inverse_feature_permutation_.begin(), inverse_feature_permutation_.end(), invalid_id);
        std::fill(materialized_feature_to_block_.begin(), materialized_feature_to_block_.end(), invalid_id);
        std::fill(materialized_feature_to_local_.begin(), materialized_feature_to_local_.end(), invalid_id);
        for (u32 execution = 0u; execution < feature_count(); ++execution) {
            inverse_feature_permutation_[feature_permutation_[execution]] = execution;
        }
        for (u32 block = 0u; block + 1u < feature_block_offsets_.size(); ++block) {
            for (u32 execution = feature_block_offsets_[block]; execution < feature_block_offsets_[block + 1u]; ++execution) {
                const u32 canonical = feature_permutation_[execution];
                materialized_feature_to_block_[canonical] = block;
                materialized_feature_to_local_[canonical] = execution - feature_block_offsets_[block];
            }
        }
        row_group_offsets_.clear();
        row_group_offsets_.push_back(0u);
        for (u64 boundary = row_group_width_; boundary < row_count_; boundary += row_group_width_) {
            row_group_offsets_.push_back(static_cast<u32>(boundary));
        }
        row_group_offsets_.push_back(row_count_);
        dirty_ = false;
        return validate();
    }

    validation_result view(packing_plan_view *out) const {
        if (out == nullptr) return validation_error(validation_code::null_pointer, invalid_id,
            "optimizer plan view output is null");
        if (dirty_) return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "optimizer execution geometry is dirty");
        packing_plan_view result;
        result.row_count = row_count_;
        result.feature_count = feature_count();
        result.feature_permutation = feature_permutation_.data();
        result.inverse_feature_permutation = inverse_feature_permutation_.data();
        result.row_group_count = static_cast<u32>(row_group_offsets_.size() - 1u);
        result.row_group_offsets = row_group_offsets_.data();
        result.feature_block_count = static_cast<u32>(feature_block_offsets_.size() - 1u);
        result.feature_block_offsets = feature_block_offsets_.data();
        const validation_result status = validate_packing_plan_view(result);
        if (!status) return status;
        *out = result;
        return validation_ok();
    }

    validation_result validate() const {
        if (support_.feature_count == 0u || blocks_.size() != support_.feature_count
            || feature_to_slot_.size() != support_.feature_count
            || members_.size() != static_cast<std::size_t>(feature_count()) * maximum_block_width_
            || union_words_.size() != static_cast<std::size_t>(feature_count()) * support_.words_per_feature) {
            return validation_error(validation_code::invalid_plan_geometry, invalid_id,
                "optimizer state dimensions disagree");
        }
        u64 total_members = 0u;
        u32 active = 0u;
        for (u32 slot = 0u; slot < blocks_.size(); ++slot) {
            const validation_result status = validate_block(slot);
            if (!status) return status;
            if (block_active(slot)) {
                ++active;
                total_members += block_width(slot);
            }
        }
        if (active == 0u || total_members != feature_count()) {
            return validation_error(validation_code::invalid_plan_geometry, invalid_id,
                "optimizer state does not cover every feature exactly once");
        }
        for (u32 feature = 0u; feature < feature_count(); ++feature) {
            const u32 slot = feature_to_slot_[feature];
            if (!block_active(slot)
                || !std::binary_search(member_row(slot), member_row(slot) + block_width(slot), feature)) {
                return validation_error(validation_code::invalid_plan_geometry, feature,
                    "feature coverage or block lookup invariant failed");
            }
        }
        if (!dirty_ && (feature_permutation_.size() != feature_count()
            || inverse_feature_permutation_.size() != feature_count()
            || feature_block_offsets_.size() != static_cast<std::size_t>(active) + 1u
            || materialized_feature_to_block_.size() != feature_count()
            || materialized_feature_to_local_.size() != feature_count()
            || row_group_offsets_.size() < 2u)) {
            return validation_error(validation_code::invalid_plan_geometry, invalid_id,
                "materialized optimizer caches are incomplete");
        }
        return validation_ok();
    }

    const std::vector<u32> &feature_permutation() const noexcept { return feature_permutation_; }
    const std::vector<u32> &inverse_feature_permutation() const noexcept { return inverse_feature_permutation_; }
    const std::vector<u32> &feature_block_offsets() const noexcept { return feature_block_offsets_; }
    const std::vector<u32> &feature_to_block() const noexcept { return materialized_feature_to_block_; }
    const std::vector<u32> &feature_to_local() const noexcept { return materialized_feature_to_local_; }
    const std::vector<u32> &row_group_offsets() const noexcept { return row_group_offsets_; }

    std::size_t estimated_additional_bytes() const noexcept {
        return blocks_.capacity() * sizeof(optimizer_block_desc)
            + members_.capacity() * sizeof(u32) + union_words_.capacity() * sizeof(u32)
            + feature_to_slot_.capacity() * sizeof(u32) + merge_scratch_.capacity() * sizeof(u32)
            + active_slots_.capacity() * sizeof(u32) + feature_permutation_.capacity() * sizeof(u32)
            + inverse_feature_permutation_.capacity() * sizeof(u32)
            + feature_block_offsets_.capacity() * sizeof(u32)
            + materialized_feature_to_block_.capacity() * sizeof(u32)
            + materialized_feature_to_local_.capacity() * sizeof(u32)
            + row_group_offsets_.capacity() * sizeof(u32);
    }

private:
    sampled_feature_support_view support_{};
    u32 row_count_ = 0u;
    u32 maximum_block_width_ = 0u;
    u32 row_group_width_ = 0u;
    std::vector<optimizer_block_desc> blocks_;
    std::vector<u32> members_;
    mutable std::vector<u32> union_words_;
    std::vector<u32> feature_to_slot_;
    std::vector<u32> merge_scratch_;
    bool dirty_ = true;
    std::vector<u32> active_slots_;
    std::vector<u32> feature_permutation_;
    std::vector<u32> inverse_feature_permutation_;
    std::vector<u32> feature_block_offsets_;
    std::vector<u32> materialized_feature_to_block_;
    std::vector<u32> materialized_feature_to_local_;
    std::vector<u32> row_group_offsets_;

    u32 *member_row(u32 slot) noexcept {
        return members_.data() + static_cast<std::size_t>(slot) * maximum_block_width_;
    }
    const u32 *member_row(u32 slot) const noexcept {
        return members_.data() + static_cast<std::size_t>(slot) * maximum_block_width_;
    }
    u32 *union_row(u32 slot) const noexcept {
        return union_words_.data() + static_cast<std::size_t>(slot) * support_.words_per_feature;
    }
    void reserve_materialization() {
        active_slots_.reserve(feature_count());
        feature_permutation_.reserve(feature_count());
        inverse_feature_permutation_.resize(feature_count(), invalid_id);
        feature_block_offsets_.reserve(static_cast<std::size_t>(feature_count()) + 1u);
        materialized_feature_to_block_.resize(feature_count(), invalid_id);
        materialized_feature_to_local_.resize(feature_count(), invalid_id);
        const std::size_t row_groups = row_count_ == 0u
            ? 1u : 1u + (static_cast<std::size_t>(row_count_ - 1u) / row_group_width_);
        row_group_offsets_.reserve(row_groups + 1u);
    }
    bool legal_distinct_blocks(u32 lhs, u32 rhs) const noexcept {
        return lhs != rhs && block_active(lhs) && block_active(rhs);
    }
    void clear_member_tail(u32 slot, u32 begin) noexcept {
        std::fill(member_row(slot) + begin, member_row(slot) + maximum_block_width_, invalid_id);
    }
    void erase_member(u32 slot, u32 feature) noexcept {
        optimizer_block_desc &block = blocks_[slot];
        u32 *members = member_row(slot);
        u32 *position = std::lower_bound(members, members + block.member_count, feature);
        std::move(position + 1u, members + block.member_count, position);
        --block.member_count;
        members[block.member_count] = invalid_id;
    }
    void insert_member(u32 slot, u32 feature) noexcept {
        optimizer_block_desc &block = blocks_[slot];
        u32 *members = member_row(slot);
        u32 *position = std::lower_bound(members, members + block.member_count, feature);
        std::move_backward(position, members + block.member_count, members + block.member_count + 1u);
        *position = feature;
        ++block.member_count;
    }
    void replace_member(u32 slot, u32 old_feature, u32 new_feature) noexcept {
        optimizer_block_desc &block = blocks_[slot];
        u32 *members = member_row(slot);
        *std::lower_bound(members, members + block.member_count, old_feature) = new_feature;
        std::sort(members, members + block.member_count);
    }
    void invalidate_union(u32 slot) noexcept {
        blocks_[slot].union_valid = 0u;
        blocks_[slot].union_count = 0u;
    }
    static u64 popcount(u32 word) noexcept { return static_cast<u64>(__builtin_popcount(word)); }
    u32 mask_tail(u32 word_index, u32 word) const noexcept {
        if (word_index + 1u != support_.words_per_feature) return word;
        const u32 tail = support_.sampled_row_count % 32u;
        return tail == 0u ? word : word & ((u32{1u} << tail) - 1u);
    }
    u32 source_word(u32 feature, u32 word) const noexcept {
        return mask_tail(word, support_.support_words[
            static_cast<std::size_t>(feature) * support_.words_per_feature + word]);
    }
    u32 union_word(u32 slot, u32 word) const noexcept {
        return blocks_[slot].member_count == 1u
            ? source_word(member_row(slot)[0], word) : union_row(slot)[word];
    }
    void ensure_union(u32 slot) const noexcept {
        optimizer_block_desc &block = const_cast<optimizer_block_desc &>(blocks_[slot]);
        if (block.union_valid != 0u) return;
        block.union_count = 0u;
        if (block.member_count == 1u) {
            for (u32 word = 0u; word < support_.words_per_feature; ++word) {
                block.union_count += popcount(source_word(member_row(slot)[0], word));
            }
            block.union_valid = 1u;
            return;
        }
        u32 *cache = union_row(slot);
        std::fill(cache, cache + support_.words_per_feature, 0u);
        for (u32 index = 0u; index < block.member_count; ++index) {
            for (u32 word = 0u; word < support_.words_per_feature; ++word) {
                cache[word] |= source_word(member_row(slot)[index], word);
            }
        }
        for (u32 word = 0u; word < support_.words_per_feature; ++word) {
            cache[word] = mask_tail(word, cache[word]);
            block.union_count += popcount(cache[word]);
        }
        block.union_valid = 1u;
    }
    u64 union_count_modified(u32 slot, u32 removed, u32 added) const noexcept {
        const optimizer_block_desc &block = blocks_[slot];
        u64 count = 0u;
        for (u32 word = 0u; word < support_.words_per_feature; ++word) {
            u32 combined = added == invalid_id ? 0u : source_word(added, word);
            for (u32 index = 0u; index < block.member_count; ++index) {
                const u32 feature = member_row(slot)[index];
                if (feature != removed) combined |= source_word(feature, word);
            }
            count += popcount(mask_tail(word, combined));
        }
        return count;
    }
    static std::int64_t signed_difference(u64 before, u64 after) noexcept {
        return before >= after ? static_cast<std::int64_t>(before - after)
            : -static_cast<std::int64_t>(after - before);
    }
    validation_result validate_block(u32 slot) const {
        const optimizer_block_desc &block = blocks_[slot];
        if (block.active == 0u) {
            if (block.member_count != 0u || block.stable_key != invalid_id) {
                return validation_error(validation_code::invalid_plan_geometry, slot,
                    "inactive optimizer block retains membership");
            }
            return validation_ok();
        }
        const u32 *members = member_row(slot);
        if (block.member_count == 0u || block.member_count > maximum_block_width_
            || block.stable_key != members[0]
            || !std::is_sorted(members, members + block.member_count)) {
            return validation_error(validation_code::invalid_plan_geometry, slot,
                "active optimizer block invariant failed");
        }
        for (u32 index = 0u; index < block.member_count; ++index) {
            const u32 feature = members[index];
            if (feature >= feature_count() || feature_to_slot_[feature] != slot
                || (index != 0u && members[index - 1u] == feature)) {
                return validation_error(validation_code::invalid_plan_geometry, feature,
                    "optimizer block member mapping is invalid");
            }
        }
        return validation_ok();
    }
    validation_result validate_touched(u32 lhs, u32 rhs) const {
        const validation_result left = validate_block(lhs);
        return left ? validate_block(rhs) : left;
    }
};

} // namespace cellpack::detail
