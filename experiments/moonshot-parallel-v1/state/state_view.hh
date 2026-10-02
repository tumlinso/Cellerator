#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace cellerator::experimental::moonshot::state {
// Host experimental views. Native adapters must use CE owner IDs/generations.
struct coordinate {
    std::uint64_t actor;
    std::uint32_t local_slot, incarnation;
};
inline bool operator==(coordinate a, coordinate b) {
    return a.actor == b.actor && a.local_slot == b.local_slot && a.incarnation == b.incarnation;
}
struct epoch {
    std::uint64_t structure, values, activity, parameters;
};
inline bool operator==(epoch a, epoch b) {
    return a.structure == b.structure && a.values == b.values &&
           a.activity == b.activity && a.parameters == b.parameters;
}
enum class activity : std::uint8_t { unknown, inactive, active };
struct binding {
    coordinate logical;
    std::size_t canonical_offset;
    bool support, capacity;
    activity runtime_activity;
    bool residual;
};
template<class T> struct owner_view {
    const T* values;
    const coordinate* coordinates;
    std::size_t count;
    const epoch* current_epoch;
};

template<class T> class state_view {
    owner_view<T> owner_;
    const binding* bindings_;
    std::size_t count_;
    epoch saved_;
public:
    state_view(owner_view<T> owner, const binding* bindings, std::size_t count)
        : owner_(owner), bindings_(bindings), count_(count), saved_{} {
        if (!owner.current_epoch) throw std::invalid_argument("missing owner epoch");
        saved_ = *owner.current_epoch;
        validate();
    }
    void validate() const {
        if (!(*owner_.current_epoch == saved_)) throw std::invalid_argument("stale saved generation");
        if ((owner_.count && (!owner_.values || !owner_.coordinates)) || (count_ && !bindings_))
            throw std::invalid_argument("missing borrowed storage");
        for (std::size_t i = 0; i < owner_.count; ++i)
            for (std::size_t j = 0; j < i; ++j)
                if (owner_.coordinates[i].actor == owner_.coordinates[j].actor &&
                    owner_.coordinates[i].local_slot == owner_.coordinates[j].local_slot)
                    throw std::invalid_argument("multiple owners/incarnations of one slot");
        for (std::size_t i = 0; i < count_; ++i) {
            const auto& b = bindings_[i];
            if (b.canonical_offset >= owner_.count || !(owner_.coordinates[b.canonical_offset] == b.logical))
                throw std::invalid_argument("stale incarnation or invalid physical alias");
            if ((b.support && !b.capacity) || (b.runtime_activity == activity::active && !b.support))
                throw std::invalid_argument("invalid support/activity/capacity");
            for (std::size_t j = 0; j < i; ++j) {
                const auto& previous = bindings_[j];
                if (previous.logical == b.logical &&
                    (previous.support != b.support || previous.capacity != b.capacity ||
                     previous.runtime_activity != b.runtime_activity))
                    throw std::invalid_argument("aliases disagree on logical flags");
            }
        }
    }
    std::vector<T> gather() const {
        validate();
        std::vector<T> out(count_);
        for (std::size_t i = 0; i < count_; ++i) out[i] = owner_.values[bindings_[i].canonical_offset];
        return out;
    }
    std::vector<T> pullback(const T* cotangents, std::size_t count) const {
        validate();
        if (count != count_ || (count && !cotangents)) throw std::invalid_argument("cotangent extent mismatch");
        std::vector<T> out(owner_.count, T{});
        for (std::size_t i = 0; i < count_; ++i) out[bindings_[i].canonical_offset] += cotangents[i];
        return out;
    }
};
} // namespace cellerator::experimental::moonshot::state
