#pragma once

#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace cellerator::modeling::cuda {

struct RegulatoryRule {
    std::uint64_t id;
    // Caller-defined bit positions, not biological IDs or an implicit ontology.
    std::vector<std::uint64_t> required_bits;
    std::vector<std::uint64_t> forbidden_bits;
};
struct RegulatoryMaskWord {
    std::uint64_t word;
    std::uint64_t required;
    std::uint64_t forbidden;
};

// Immutable host preparation of a shared sparse word-mask bank. Rules are in
// canonical stable-ID order. Only words touched by a rule are stored.
class PackedRegulatoryRules final {
public:
    PackedRegulatoryRules(std::vector<RegulatoryRule> rules,
                          std::uint64_t input_bit_count,
                          std::uint64_t structure_epoch = 0)
        : input_bits_(input_bit_count), epoch_(structure_epoch) {
        std::sort(rules.begin(), rules.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
        offsets_.push_back(0);
        for (const auto& rule : rules) {
            if (!ids_.empty() && ids_.back() == rule.id)
                throw std::invalid_argument("duplicate regulatory rule ID");
            std::map<std::uint64_t, RegulatoryMaskWord> words;
            const auto add = [&](std::uint64_t bit, bool required) {
                if (bit >= input_bits_) throw std::out_of_range("regulatory input bit outside declared domain");
                auto& word = words[bit / 64];
                word.word = bit / 64;
                const auto mask = std::uint64_t{1} << (bit % 64);
                if ((word.required | word.forbidden) & mask)
                    throw std::invalid_argument("duplicate or contradictory regulatory input");
                if (required) word.required |= mask;
                else word.forbidden |= mask;
            };
            for (auto bit : rule.required_bits) add(bit, true);
            for (auto bit : rule.forbidden_bits) add(bit, false);
            for (const auto& item : words) masks_.push_back(item.second);
            ids_.push_back(rule.id);
            offsets_.push_back(masks_.size());
        }
    }
    std::uint64_t input_bit_count() const noexcept { return input_bits_; }
    std::uint64_t structure_epoch() const noexcept { return epoch_; }
    const std::vector<std::uint64_t>& rule_ids() const noexcept { return ids_; }
    const std::vector<std::uint64_t>& offsets() const noexcept { return offsets_; }
    const std::vector<RegulatoryMaskWord>& masks() const noexcept { return masks_; }
    std::size_t rule_ordinal(std::uint64_t id) const {
        const auto p = std::lower_bound(ids_.begin(), ids_.end(), id);
        if (p == ids_.end() || *p != id) throw std::out_of_range("unknown regulatory rule ID");
        return static_cast<std::size_t>(p - ids_.begin());
    }
private:
    std::uint64_t input_bits_, epoch_;
    std::vector<std::uint64_t> ids_, offsets_;
    std::vector<RegulatoryMaskWord> masks_;
};

// Private host bits for one instance; copy explicitly to inherit a snapshot.
// Accessibility controls a rule's modeled permission, not physical residency.
class PackedRegulatoryState final {
public:
    PackedRegulatoryState(std::uint64_t input_bits, std::uint64_t rule_count,
                          bool initially_accessible = true)
        : input_bits_(input_bits), rule_count_(rule_count),
          inputs_(words(input_bits), 0), accessible_(words(rule_count), initially_accessible ? ~0ULL : 0) {
        if (rule_count % 64 && !accessible_.empty())
            accessible_.back() &= (std::uint64_t{1} << (rule_count % 64)) - 1;
    }
    void set_input(std::uint64_t bit, bool value) { set(inputs_, bit, input_bits_, value); }
    void set_accessible(std::uint64_t rule, bool value) { set(accessible_, rule, rule_count_, value); }
    const std::vector<std::uint64_t>& inputs() const noexcept { return inputs_; }
    const std::vector<std::uint64_t>& accessibility() const noexcept { return accessible_; }
    static std::uint64_t words(std::uint64_t bits) noexcept { return bits / 64 + (bits % 64 != 0); }
private:
    static void set(std::vector<std::uint64_t>& storage, std::uint64_t bit,
                    std::uint64_t count, bool value) {
        if (bit >= count) throw std::out_of_range("regulatory state bit outside declared domain");
        const auto mask = std::uint64_t{1} << (bit % 64);
        if (value) storage[bit / 64] |= mask;
        else storage[bit / 64] &= ~mask;
    }
    std::uint64_t input_bits_, rule_count_;
    std::vector<std::uint64_t> inputs_, accessible_;
};

struct RegulatoryRuleBankView {
    const std::uint64_t* offsets = nullptr; // rule_count + 1
    const RegulatoryMaskWord* masks = nullptr;
    std::uint64_t rule_count = 0, mask_count = 0, input_bit_count = 0;
};
struct RegulatoryStateView {
    const std::uint64_t* inputs = nullptr;
    const std::uint64_t* accessibility = nullptr;
    std::uint64_t instance_count = 0;
    std::uint64_t input_word_stride = 0, accessibility_word_stride = 0;
};
struct RegulatoryOccurrence {
    std::uint64_t instance;
    std::uint64_t rule; // dense ordinal in the retained shared rule bank
};
enum RegulatoryGateFlag : std::uint32_t {
    regulatory_gate_success = 0,
    regulatory_gate_invalid_occurrence = 1,
    regulatory_gate_invalid_rule = 2,
    regulatory_gate_insufficient_capacity = 4,
};
struct RegulatoryGateStatus {
    std::uint64_t eligible_count = 0; // exact unless an input-content error occurs
    std::uint64_t compact_written = 0; // zero on any flag, or when compaction is off
    std::uint32_t flags = 0;
    std::uint32_t reserved = 0;
};
struct RegulatoryGateOutput {
    std::uint64_t* eligibility_bits = nullptr; // indexed by worklist occurrence
    std::uint64_t eligibility_word_capacity = 0;
    std::uint64_t* compact_occurrence_indices = nullptr;
    std::uint64_t compact_capacity = 0;
    RegulatoryGateStatus* status = nullptr;
    bool compact = false;
};

// Evaluate ONLY the supplied (instance,rule) occurrences. Eligibility is
// accessible && all(required bits active) && none(forbidden bits active).
// Duplicate occurrences remain distinct; compact indices have unspecified order.
// All pointers borrow disjoint device memory on the current device. Caller owns
// lifetime through stream completion. No allocation, sync or device switch.
// Check CUDA return then device status after completion. On ANY status flag,
// neither output is publishable (storage can have partial writes). Overflow
// reports the exact eligible_count if there are no input errors; retry safely.
cudaError_t evaluate_regulatory_gates(RegulatoryRuleBankView bank,
                                      RegulatoryStateView states,
                                      const RegulatoryOccurrence* worklist,
                                      std::uint64_t occurrence_count,
                                      RegulatoryGateOutput output,
                                      cudaStream_t stream) noexcept;

// Experimental alternative representation: one bit per independent cell,
// 32 cells per word. Features are planes [feature][instance_word]. No automatic
// conversion or dispatch connects this entrypoint to the occurrence API above.
struct RegulatoryBitplaneRules {
    const std::uint64_t* required_offsets = nullptr; // rule_count + 1
    const std::uint64_t* forbidden_offsets = nullptr;
    const std::uint64_t* required_features = nullptr;
    const std::uint64_t* forbidden_features = nullptr;
    std::uint64_t rule_count = 0, required_count = 0, forbidden_count = 0, feature_count = 0;
};
struct RegulatoryBitplaneState {
    const std::uint32_t* features = nullptr; // feature-major planes
    const std::uint32_t* accessibility = nullptr; // optional [rule][instance_word]; null means all open
    const std::uint32_t* valid_instances = nullptr; // mandatory per-word validity/tail masks
    std::uint64_t instance_word_count = 0;
};
struct RegulatoryBitplaneWork {
    std::uint64_t rule;
    std::uint64_t instance_word;
};
struct RegulatoryBitplaneStatus {
    std::uint64_t written = 0; // output words; zero on any content error
    std::uint32_t flags = 0; // RegulatoryGateFlag values
    std::uint32_t reserved = 0;
};
struct RegulatoryBitplaneOutput {
    std::uint32_t* eligibility_words = nullptr; // one packed result per work item
    std::uint64_t capacity = 0;
    RegulatoryBitplaneStatus* status = nullptr;
};
// AND required planes, ANDNOT forbidden planes, accessibility and validity.
// Results stay packed for subsequent Boolean transitions or caller compaction.
// Duplicate work items remain duplicate words. Inputs/outputs must not alias:
// use separate next-state storage between stream-ordered transitions. Same
// explicit stream, ownership and fail-closed publication contract as above.
cudaError_t evaluate_regulatory_bitplanes(RegulatoryBitplaneRules rules,
                                          RegulatoryBitplaneState state,
                                          const RegulatoryBitplaneWork* work,
                                          std::uint64_t work_count,
                                          RegulatoryBitplaneOutput output,
                                          cudaStream_t stream) noexcept;

} // namespace cellerator::modeling::cuda
