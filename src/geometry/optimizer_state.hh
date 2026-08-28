#pragma once

#include "Cellerator/geometry/optimizer.hh"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace cellpack::detail {

class optimizer_state {
public:
    struct block_state {
        bool active = false;
        u32 stable_key = invalid_id;
        u32 generation = 0u;
        std::vector<u32> members;
        mutable bool union_valid = false;
        mutable u64 union_count = 0u;
        mutable std::vector<u32> union_words;
    };

    optimizer_state() = default;

    validation_result initialize(
        u32 row_count,
        const sampled_feature_support_view &support,
        u32 maximum_block_width,
        u32 row_group_width) {
        if (support.feature_count == 0u || maximum_block_width == 0u || row_group_width == 0u) {
            return validation_error(validation_code::invalid_plan_geometry, invalid_id,
                "optimizer dimensions and configured widths must be nonzero");
        }
        support_ = support;
        row_count_ = row_count;
        maximum_block_width_ = maximum_block_width;
        row_group_width_ = row_group_width;
        blocks_.clear();
        blocks_.resize(support.feature_count);
        for (u32 feature = 0u; feature < support.feature_count; ++feature) {
            block_state &block = blocks_[feature];
            block.active = true;
            block.stable_key = feature;
            block.members = {feature};
        }
        rebuild_feature_lookup();
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
        return slot < blocks_.size() && blocks_[slot].active
            ? static_cast<u32>(blocks_[slot].members.size()) : 0u;
    }
    u32 block_stable_key(u32 slot) const noexcept {
        return slot < blocks_.size() && blocks_[slot].active ? blocks_[slot].stable_key : invalid_id;
    }
    u32 block_generation(u32 slot) const noexcept {
        return slot < blocks_.size() ? blocks_[slot].generation : 0u;
    }
    bool block_active(u32 slot) const noexcept {
        return slot < blocks_.size() && blocks_[slot].active;
    }
    const std::vector<u32> &block_members(u32 slot) const { return blocks_.at(slot).members; }
    u32 active_block_count() const noexcept {
        u32 result = 0u;
        for (const block_state &block : blocks_) result += block.active ? 1u : 0u;
        return result;
    }

    validation_result merge_blocks(u32 lhs_slot, u32 rhs_slot) {
        if (!legal_distinct_blocks(lhs_slot, rhs_slot)) {
            return validation_error(validation_code::invalid_plan_geometry, invalid_id, "merge requires two distinct active blocks");
        }
        if (blocks_[lhs_slot].members.size() + blocks_[rhs_slot].members.size() > maximum_block_width_) {
            return validation_error(validation_code::invalid_plan_geometry, invalid_id, "merge exceeds maximum block width");
        }
        if (blocks_[rhs_slot].stable_key < blocks_[lhs_slot].stable_key) std::swap(lhs_slot, rhs_slot);
        block_state &lhs = blocks_[lhs_slot];
        block_state &rhs = blocks_[rhs_slot];
        lhs.members.insert(lhs.members.end(), rhs.members.begin(), rhs.members.end());
        std::sort(lhs.members.begin(), lhs.members.end());
        lhs.stable_key = lhs.members.front();
        ++lhs.generation;
        invalidate_union(lhs);
        rhs.active = false;
        rhs.members.clear();
        rhs.stable_key = invalid_id;
        ++rhs.generation;
        invalidate_union(rhs);
        rebuild_feature_lookup();
        dirty_ = true;
        return validate();
    }

    validation_result move_feature(u32 feature, u32 destination_slot) {
        if (feature >= feature_count() || !block_active(destination_slot)) {
            return validation_error(validation_code::invalid_plan_geometry, feature, "move feature or destination is invalid");
        }
        const u32 source_slot = feature_to_slot_[feature];
        if (source_slot == destination_slot) {
            return validation_error(validation_code::invalid_plan_geometry, feature, "move destination equals source block");
        }
        if (blocks_[destination_slot].members.size() >= maximum_block_width_) {
            return validation_error(validation_code::invalid_plan_geometry, destination_slot, "move destination is at maximum width");
        }
        block_state &source = blocks_[source_slot];
        block_state &destination = blocks_[destination_slot];
        source.members.erase(std::lower_bound(source.members.begin(), source.members.end(), feature));
        destination.members.insert(
            std::lower_bound(destination.members.begin(), destination.members.end(), feature), feature);
        ++source.generation;
        ++destination.generation;
        invalidate_union(source);
        invalidate_union(destination);
        if (source.members.empty()) {
            source.active = false;
            source.stable_key = invalid_id;
        } else {
            source.stable_key = source.members.front();
        }
        destination.stable_key = destination.members.front();
        rebuild_feature_lookup();
        dirty_ = true;
        return validate();
    }

    validation_result swap_features(u32 feature_a, u32 feature_b) {
        if (feature_a >= feature_count() || feature_b >= feature_count() || feature_a == feature_b) {
            return validation_error(validation_code::invalid_plan_geometry, invalid_id, "swap features are invalid");
        }
        const u32 slot_a = feature_to_slot_[feature_a], slot_b = feature_to_slot_[feature_b];
        if (slot_a == slot_b) {
            return validation_error(validation_code::invalid_plan_geometry, feature_a, "swap features already share a block");
        }
        block_state &block_a = blocks_[slot_a];
        block_state &block_b = blocks_[slot_b];
        *std::lower_bound(block_a.members.begin(), block_a.members.end(), feature_a) = feature_b;
        *std::lower_bound(block_b.members.begin(), block_b.members.end(), feature_b) = feature_a;
        std::sort(block_a.members.begin(), block_a.members.end());
        std::sort(block_b.members.begin(), block_b.members.end());
        block_a.stable_key = block_a.members.front();
        block_b.stable_key = block_b.members.front();
        ++block_a.generation;
        ++block_b.generation;
        invalidate_union(block_a);
        invalidate_union(block_b);
        rebuild_feature_lookup();
        dirty_ = true;
        return validate();
    }

    std::int64_t merge_proxy_gain(u32 lhs_slot, u32 rhs_slot) const {
        if (!legal_distinct_blocks(lhs_slot, rhs_slot)) return std::numeric_limits<std::int64_t>::min();
        ensure_union(lhs_slot);
        ensure_union(rhs_slot);
        u64 intersection = 0u;
        for (u32 word = 0u; word < support_.words_per_feature; ++word) {
            intersection += popcount(mask_tail(word, union_word(lhs_slot, word) & union_word(rhs_slot, word)));
        }
        return static_cast<std::int64_t>(intersection);
    }

    std::int64_t move_proxy_gain(u32 feature, u32 destination_slot) const {
        if (feature >= feature_count() || !block_active(destination_slot)) return std::numeric_limits<std::int64_t>::min();
        const u32 source_slot = feature_to_slot_[feature];
        if (source_slot == destination_slot || blocks_[destination_slot].members.size() >= maximum_block_width_) {
            return std::numeric_limits<std::int64_t>::min();
        }
        ensure_union(source_slot);
        ensure_union(destination_slot);
        const u64 before = blocks_[source_slot].union_count + blocks_[destination_slot].union_count;
        std::vector<u32> source_members = blocks_[source_slot].members;
        source_members.erase(std::lower_bound(source_members.begin(), source_members.end(), feature));
        std::vector<u32> destination_members = blocks_[destination_slot].members;
        destination_members.insert(std::lower_bound(destination_members.begin(), destination_members.end(), feature), feature);
        const u64 after = union_count_for_members(source_members) + union_count_for_members(destination_members);
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
        std::vector<u32> members_a = blocks_[slot_a].members;
        std::vector<u32> members_b = blocks_[slot_b].members;
        *std::lower_bound(members_a.begin(), members_a.end(), feature_a) = feature_b;
        *std::lower_bound(members_b.begin(), members_b.end(), feature_b) = feature_a;
        std::sort(members_a.begin(), members_a.end());
        std::sort(members_b.begin(), members_b.end());
        const u64 after = union_count_for_members(members_a) + union_count_for_members(members_b);
        return signed_difference(before, after);
    }

    validation_result materialize_execution_geometry() {
        const validation_result state_status = validate();
        if (!state_status) return state_status;
        std::vector<u32> active_slots;
        active_slots.reserve(blocks_.size());
        for (u32 slot = 0u; slot < blocks_.size(); ++slot) {
            if (blocks_[slot].active) active_slots.push_back(slot);
        }
        std::sort(active_slots.begin(), active_slots.end(), [&](u32 lhs, u32 rhs) {
            return blocks_[lhs].stable_key < blocks_[rhs].stable_key;
        });
        feature_permutation_.clear();
        feature_permutation_.reserve(feature_count());
        feature_block_offsets_.assign(1u, 0u);
        for (u32 slot : active_slots) {
            feature_permutation_.insert(feature_permutation_.end(),
                blocks_[slot].members.begin(), blocks_[slot].members.end());
            feature_block_offsets_.push_back(static_cast<u32>(feature_permutation_.size()));
        }
        inverse_feature_permutation_.assign(feature_count(), invalid_id);
        materialized_feature_to_block_.assign(feature_count(), invalid_id);
        materialized_feature_to_local_.assign(feature_count(), invalid_id);
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
        row_group_offsets_.assign(1u, 0u);
        for (u64 boundary = row_group_width_; boundary < row_count_; boundary += row_group_width_) {
            row_group_offsets_.push_back(static_cast<u32>(boundary));
        }
        row_group_offsets_.push_back(row_count_);
        dirty_ = false;
        return validate();
    }

    validation_result view(packing_plan_view *out) const {
        if (out == nullptr) return validation_error(validation_code::null_pointer, invalid_id, "optimizer plan view output is null");
        if (dirty_) return validation_error(validation_code::invalid_plan_geometry, invalid_id, "optimizer execution geometry is dirty");
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
            || feature_to_slot_.size() != support_.feature_count) {
            return validation_error(validation_code::invalid_plan_geometry, invalid_id, "optimizer state dimensions disagree");
        }
        std::vector<u32> seen(feature_count(), 0u);
        u32 active = 0u;
        for (u32 slot = 0u; slot < blocks_.size(); ++slot) {
            const block_state &block = blocks_[slot];
            if (!block.active) {
                if (!block.members.empty()) return validation_error(validation_code::invalid_plan_geometry, slot, "inactive block owns members");
                continue;
            }
            ++active;
            if (block.members.empty() || block.members.size() > maximum_block_width_
                || block.stable_key != block.members.front()
                || !std::is_sorted(block.members.begin(), block.members.end())) {
                return validation_error(validation_code::invalid_plan_geometry, slot, "active block invariant failed");
            }
            for (u32 feature : block.members) {
                if (feature >= feature_count() || ++seen[feature] != 1u || feature_to_slot_[feature] != slot) {
                    return validation_error(validation_code::invalid_plan_geometry, feature, "feature coverage or block lookup invariant failed");
                }
            }
            if (block.union_valid && block.members.size() > 1u
                && block.union_words.size() != support_.words_per_feature) {
                return validation_error(validation_code::invalid_plan_geometry, slot, "block support cache shape is invalid");
            }
        }
        if (active == 0u) return validation_error(validation_code::invalid_plan_geometry, invalid_id, "optimizer state has no active blocks");
        for (u32 count : seen) if (count != 1u) {
            return validation_error(validation_code::invalid_plan_geometry, invalid_id, "optimizer state does not cover every feature exactly once");
        }
        if (!dirty_) {
            if (feature_permutation_.size() != feature_count()
                || inverse_feature_permutation_.size() != feature_count()
                || feature_block_offsets_.size() != static_cast<std::size_t>(active) + 1u
                || materialized_feature_to_block_.size() != feature_count()
                || materialized_feature_to_local_.size() != feature_count()
                || row_group_offsets_.size() < 2u) {
                return validation_error(validation_code::invalid_plan_geometry, invalid_id, "materialized optimizer caches are incomplete");
            }
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
        std::size_t bytes = feature_to_slot_.capacity() * sizeof(u32);
        bytes += feature_permutation_.capacity() * sizeof(u32)
            + inverse_feature_permutation_.capacity() * sizeof(u32)
            + feature_block_offsets_.capacity() * sizeof(u32)
            + materialized_feature_to_block_.capacity() * sizeof(u32)
            + materialized_feature_to_local_.capacity() * sizeof(u32)
            + row_group_offsets_.capacity() * sizeof(u32);
        for (const block_state &block : blocks_) {
            bytes += sizeof(block_state) + block.members.capacity() * sizeof(u32)
                + block.union_words.capacity() * sizeof(u32);
        }
        return bytes;
    }

private:
    sampled_feature_support_view support_{};
    u32 row_count_ = 0u;
    u32 maximum_block_width_ = 0u;
    u32 row_group_width_ = 0u;
    std::vector<block_state> blocks_;
    std::vector<u32> feature_to_slot_;
    bool dirty_ = true;
    std::vector<u32> feature_permutation_;
    std::vector<u32> inverse_feature_permutation_;
    std::vector<u32> feature_block_offsets_;
    std::vector<u32> materialized_feature_to_block_;
    std::vector<u32> materialized_feature_to_local_;
    std::vector<u32> row_group_offsets_;

    bool legal_distinct_blocks(u32 lhs, u32 rhs) const noexcept {
        return lhs != rhs && block_active(lhs) && block_active(rhs);
    }
    static void invalidate_union(block_state &block) {
        block.union_valid = false;
        block.union_count = 0u;
        block.union_words.clear();
    }
    static u64 popcount(u32 word) noexcept {
        return static_cast<u64>(__builtin_popcount(word));
    }
    u32 mask_tail(u32 word_index, u32 word) const noexcept {
        if (word_index + 1u != support_.words_per_feature) return word;
        const u32 tail = support_.sampled_row_count % 32u;
        if (tail == 0u) return word;
        return word & ((u32{1u} << tail) - 1u);
    }
    u32 source_word(u32 feature, u32 word) const noexcept {
        return mask_tail(word, support_.support_words[
            static_cast<std::size_t>(feature) * support_.words_per_feature + word]);
    }
    u32 union_word(u32 slot, u32 word) const noexcept {
        const block_state &block = blocks_[slot];
        return block.members.size() == 1u ? source_word(block.members.front(), word) : block.union_words[word];
    }
    void ensure_union(u32 slot) const {
        block_state &block = const_cast<block_state &>(blocks_[slot]);
        if (block.union_valid) return;
        if (block.members.size() == 1u) {
            block.union_count = union_count_for_members(block.members);
            block.union_words.clear();
            block.union_valid = true;
            return;
        }
        block.union_words.assign(support_.words_per_feature, 0u);
        for (u32 feature : block.members) {
            for (u32 word = 0u; word < support_.words_per_feature; ++word) {
                block.union_words[word] |= source_word(feature, word);
            }
        }
        block.union_count = 0u;
        for (u32 word = 0u; word < support_.words_per_feature; ++word) {
            block.union_words[word] = mask_tail(word, block.union_words[word]);
            block.union_count += popcount(block.union_words[word]);
        }
        block.union_valid = true;
    }
    u64 union_count_for_members(const std::vector<u32> &members) const {
        if (members.empty() || support_.words_per_feature == 0u) return 0u;
        u64 count = 0u;
        for (u32 word = 0u; word < support_.words_per_feature; ++word) {
            u32 combined = 0u;
            for (u32 feature : members) combined |= source_word(feature, word);
            count += popcount(mask_tail(word, combined));
        }
        return count;
    }
    static std::int64_t signed_difference(u64 before, u64 after) noexcept {
        if (before >= after) return static_cast<std::int64_t>(before - after);
        return -static_cast<std::int64_t>(after - before);
    }
    void rebuild_feature_lookup() {
        feature_to_slot_.assign(feature_count(), invalid_id);
        for (u32 slot = 0u; slot < blocks_.size(); ++slot) {
            if (!blocks_[slot].active) continue;
            for (u32 feature : blocks_[slot].members) feature_to_slot_[feature] = slot;
        }
    }
};

} // namespace cellpack::detail
