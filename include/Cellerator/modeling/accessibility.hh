#pragma once

// Persistent, branchable state for caller-defined accessibility programs.
// Program structure is immutable and shared; open/closed and input state use
// copy-on-write word bitsets. This records eligibility, never learned routing.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cellerator::modeling {

template<class Id>
struct AccessibilityProgram {
    Id id;
    std::vector<Id> required_inputs;
    std::vector<Id> forbidden_inputs;
    std::uint64_t revision = 0;
    bool initially_open = true;
};

enum class AccessibilityEntity : std::uint8_t { program, input };

template<class Id>
struct AccessibilityChange {
    AccessibilityEntity entity;
    Id id;
    bool enabled;
    std::uint64_t generation;
    std::string provenance;
};

template<class Id>
struct AccessibilityResult {
    bool eligible;
    std::uint64_t structure_revision;
    std::uint64_t program_revision;
    std::uint64_t state_generation;
};

// A gate over persistent inherited input accessibility and a model-defined
// program. Both bitsets are compact; a program query performs wordwise masks.
template<class Id>
class AccessibilityState {
    static_assert(std::is_copy_constructible_v<Id>, "IDs must be copy constructible");

    struct ProgramBits {
        std::vector<std::uint64_t> required;
        std::vector<std::uint64_t> forbidden;
        std::uint64_t revision;
        bool initially_open;
    };
    struct Structure {
        std::uint64_t revision;
        std::vector<Id> programs;
        std::vector<Id> inputs;
        std::unordered_map<Id, std::size_t> program_index;
        std::unordered_map<Id, std::size_t> input_index;
        std::vector<ProgramBits> bits;
    };
    struct MutableState {
        std::vector<std::uint64_t> open;
        std::vector<std::uint64_t> active_inputs;
        std::vector<AccessibilityChange<Id>> changes;
        std::uint64_t generation = 0;
    };

public:
    explicit AccessibilityState(std::vector<AccessibilityProgram<Id>> programs,
                                std::uint64_t structure_revision = 0)
        : structure_(make_structure(std::move(programs), structure_revision)),
          state_(std::make_shared<MutableState>()) {
        state_->open.assign(word_count(structure_->programs.size()), 0);
        state_->active_inputs.assign(word_count(structure_->inputs.size()), 0);
        for (std::size_t i = 0; i < structure_->programs.size(); ++i)
            if (structure_->bits[i].initially_open) set_bit(state_->open, i, true);
    }

    // A fork inherits the current flags and input mask. Its next write clones
    // those small bitsets; program/incidence structure stays shared.
    AccessibilityState fork() const {
        return AccessibilityState(structure_, state_);
    }

    bool shares_structure_with(const AccessibilityState &other) const noexcept {
        return structure_ == other.structure_;
    }

    std::uint64_t structure_revision() const noexcept { return structure_->revision; }
    std::uint64_t generation() const noexcept { return state_->generation; }
    std::size_t program_count() const noexcept { return structure_->programs.size(); }
    std::size_t input_count() const noexcept { return structure_->inputs.size(); }
    std::size_t storage_bytes() const noexcept {
        return sizeof(std::uint64_t) *
            (state_->open.size() + state_->active_inputs.size());
    }

    bool is_open(const Id &program) const {
        return bit(state_->open, program_position(program));
    }

    bool input_active(const Id &input) const {
        return bit(state_->active_inputs, input_position(input));
    }

    bool set_open(const Id &program, bool open, std::string provenance = {}) {
        const auto index = program_position(program);
        if (bit(state_->open, index) == open) return false;
        detach();
        set_bit(state_->open, index, open);
        record(AccessibilityEntity::program, program, open, std::move(provenance));
        return true;
    }

    bool set_input(const Id &input, bool active, std::string provenance = {}) {
        const auto index = input_position(input);
        if (bit(state_->active_inputs, index) == active) return false;
        detach();
        set_bit(state_->active_inputs, index, active);
        record(AccessibilityEntity::input, input, active, std::move(provenance));
        return true;
    }

    // Validate the entire update before changing any bit. A batch advances the
    // generation once while preserving per-input provenance at that generation.
    std::size_t set_inputs(const std::vector<std::pair<Id, bool>> &updates,
                           std::string provenance = {}) {
        std::unordered_set<Id> seen;
        std::vector<std::pair<std::size_t, bool>> changed;
        changed.reserve(updates.size());
        for (const auto &[id, enabled] : updates) {
            if (!seen.insert(id).second)
                throw std::invalid_argument("duplicate input in accessibility update");
            const auto index = input_position(id);
            if (bit(state_->active_inputs, index) != enabled)
                changed.emplace_back(index, enabled);
        }
        if (changed.empty()) return 0;
        detach();
        ++state_->generation;
        for (std::size_t i = 0; i < changed.size(); ++i) {
            const auto &[index, enabled] = changed[i];
            set_bit(state_->active_inputs, index, enabled);
            state_->changes.push_back({AccessibilityEntity::input,
                structure_->inputs[index], enabled, state_->generation,
                provenance});
        }
        return changed.size();
    }

    AccessibilityResult<Id> query(const Id &program) const {
        const auto index = program_position(program);
        const auto &masks = structure_->bits[index];
        bool inputs_match = true;
        for (std::size_t word = 0; word < masks.required.size(); ++word) {
            const auto active = state_->active_inputs[word];
            if ((active & masks.required[word]) != masks.required[word] ||
                (active & masks.forbidden[word]) != 0) {
                inputs_match = false;
                break;
            }
        }
        return {bit(state_->open, index) && inputs_match,
                structure_->revision, masks.revision, state_->generation};
    }

    bool eligible(const Id &program) const { return query(program).eligible; }

    std::vector<Id> eligible_programs() const {
        std::vector<Id> result;
        for (const auto &id : structure_->programs)
            if (eligible(id)) result.push_back(id);
        return result;
    }

    const std::vector<AccessibilityChange<Id>> &changes() const noexcept {
        return state_->changes;
    }

private:
    AccessibilityState(std::shared_ptr<const Structure> structure,
                       std::shared_ptr<MutableState> state)
        : structure_(std::move(structure)), state_(std::move(state)) {}

    static std::size_t word_count(std::size_t bits) noexcept {
        return (bits + 63u) / 64u;
    }
    static bool bit(const std::vector<std::uint64_t> &words, std::size_t index) noexcept {
        return ((words[index / 64u] >> (index % 64u)) & 1u) != 0;
    }
    static void set_bit(std::vector<std::uint64_t> &words, std::size_t index,
                        bool value) noexcept {
        const auto mask = std::uint64_t{1} << (index % 64u);
        auto &word = words[index / 64u];
        if (value) word |= mask;
        else word &= ~mask;
    }

    static std::shared_ptr<const Structure> make_structure(
        std::vector<AccessibilityProgram<Id>> programs, std::uint64_t revision) {
        auto result = std::make_shared<Structure>();
        result->revision = revision;
        result->programs.reserve(programs.size());
        for (const auto &program : programs) {
            if (!result->program_index.emplace(program.id, result->programs.size()).second)
                throw std::invalid_argument("duplicate accessibility program ID");
            result->programs.push_back(program.id);
            for (const auto &id : program.required_inputs) {
                if (result->input_index.emplace(id, result->inputs.size()).second)
                    result->inputs.push_back(id);
            }
            for (const auto &id : program.forbidden_inputs) {
                if (result->input_index.emplace(id, result->inputs.size()).second)
                    result->inputs.push_back(id);
            }
        }
        const auto words = word_count(result->inputs.size());
        result->bits.reserve(programs.size());
        for (const auto &program : programs) {
            ProgramBits masks{std::vector<std::uint64_t>(words, 0),
                              std::vector<std::uint64_t>(words, 0),
                              program.revision, program.initially_open};
            std::unordered_set<Id> required;
            for (const auto &id : program.required_inputs) {
                if (!required.insert(id).second)
                    throw std::invalid_argument("duplicate required accessibility input");
                set_bit(masks.required, result->input_index.at(id), true);
            }
            std::unordered_set<Id> forbidden;
            for (const auto &id : program.forbidden_inputs) {
                if (!forbidden.insert(id).second)
                    throw std::invalid_argument("duplicate forbidden accessibility input");
                if (required.contains(id))
                    throw std::invalid_argument("an accessibility input cannot be both required and forbidden");
                set_bit(masks.forbidden, result->input_index.at(id), true);
            }
            result->bits.push_back(std::move(masks));
        }
        return result;
    }

    std::size_t program_position(const Id &id) const {
        const auto it = structure_->program_index.find(id);
        if (it == structure_->program_index.end())
            throw std::out_of_range("unknown accessibility program ID");
        return it->second;
    }
    std::size_t input_position(const Id &id) const {
        const auto it = structure_->input_index.find(id);
        if (it == structure_->input_index.end())
            throw std::out_of_range("unknown accessibility input ID");
        return it->second;
    }
    void detach() {
        if (!state_.unique()) state_ = std::make_shared<MutableState>(*state_);
    }
    void record(AccessibilityEntity entity, const Id &id, bool enabled,
                std::string provenance) {
        ++state_->generation;
        state_->changes.push_back({entity, id, enabled, state_->generation,
                                   std::move(provenance)});
    }

    std::shared_ptr<const Structure> structure_;
    std::shared_ptr<MutableState> state_;
};

class HysteresisThreshold {
public:
    HysteresisThreshold(double open_at, double close_at)
        : open_at_(open_at), close_at_(close_at) {
        if (!(close_at < open_at))
            throw std::invalid_argument("close threshold must be below open threshold");
    }

    bool update(bool currently_open, double signal) const {
        if (currently_open) return signal > close_at_;
        return signal >= open_at_;
    }

private:
    double open_at_;
    double close_at_;
};

} // namespace cellerator::modeling
