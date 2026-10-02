#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::experimental::moonshot {
// Prototype POD views; adapter must reuse native IDs, memory views and owner.
// Slots are LOCAL to their actor. Neither tile position nor slot is ontology.
struct coordinate_ref {
    std::uint64_t actor;
    std::uint32_t local_slot;
    std::uint32_t slot_generation;
};
struct epoch_stamp {
    std::uint64_t structure, values, activity, parameters;
};
enum class slot_lifecycle : std::uint8_t { live, reusable, reserved, retired };
// Capacity ownership, structural existence and runtime activity are distinct.
struct slot_binding {
    coordinate_ref logical;
    std::uint64_t canonical_offset;
    slot_lifecycle lifecycle;
};

template<class T, int Tile>
struct alignas(32) tile_payload {
    static_assert(Tile>0 && Tile<=32);
    T elements[Tile*Tile];
};

template<class T, int Tile>
struct cell_tile_view {
    tile_payload<T,Tile>* payloads;       // non-owning physical projection
    const coordinate_ref* coordinates;  // one mapping per stored logical entry
    const std::uint32_t* structural_words;
    const std::uint32_t* forward_work;
    const std::uint32_t* response_work;  // zero value does not prove zero response
    std::uint64_t tile_count, coordinate_count;
    std::uint64_t forward_count, response_count;
    T* canonical_values;
    std::uint64_t canonical_count;
    epoch_stamp version;
};

struct residual_entry { std::uint64_t canonical_offset; std::uint32_t destination,source; };
struct region_descriptor {
    std::uint32_t program_kind;
    std::uint32_t first_tile,tile_count;
    std::uint32_t first_residual,residual_count;
    std::uint32_t output_owner;
    std::uint32_t gate_id;
};
// A register object is ephemeral kernel state, NOT persistent state allocation.
struct promotion_boundary {
    epoch_stamp old_version,new_version;
    const std::uint64_t* old_to_new;
    std::uint64_t mapping_count;
    bool requires_closed_tapes;
};
static_assert(std::is_trivially_copyable_v<cell_tile_view<float,16>>);
static_assert(alignof(tile_payload<float,16>)>=32);
} // namespace cellerator::experimental::moonshot
