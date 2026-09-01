#pragma once

#include <cstdint>
#include <type_traits>

namespace cellerator::profiling::joint_compiler {

struct stable_identity_v1 {
    std::uint64_t low = 0u;
    std::uint64_t high = 0u;
};

struct mechanism_manifest_v1 {
    stable_identity_v1 mechanism_identity{};
    std::uint64_t candidate_id = 0u;
    std::uint64_t kernel_id = 0u;
    std::uint64_t useful_interactions = 0u;
    std::uint64_t bytes_moved = 0u;
    std::uint64_t launch_count = 0u;
    std::uint64_t preparation_nanoseconds = 0u;
    std::uint64_t execution_nanoseconds = 0u;
};

struct atom_profile_record_v1 {
    stable_identity_v1 atom_identity{};
    stable_identity_v1 atom_species_identity{};
    std::uint64_t structure_epoch = 0u;
    std::uint64_t value_generation = 0u;
    const mechanism_manifest_v1 *mechanisms = nullptr;
    std::uint64_t mechanism_count = 0u;
};

enum class export_code_v1 : std::uint8_t {
    success = 0u,
    invalid_argument,
    invalid_identity,
    duplicate_atom,
    invalid_measurement,
    sink_failed,
};

struct export_status_v1 {
    export_code_v1 code = export_code_v1::success;
    std::uint64_t atom_index = 0u;
    std::uint64_t mechanism_index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == export_code_v1::success;
    }
};

using atom_profile_sink_v1 = bool (*)(void *context,
    const atom_profile_record_v1 &record) noexcept;

struct profile_export_sink_v1 {
    void *context = nullptr;
    atom_profile_sink_v1 emit = nullptr;
};

constexpr bool valid_identity_v1(stable_identity_v1 identity) noexcept {
    return identity.low != 0u || identity.high != 0u;
}

constexpr bool same_identity_v1(
    stable_identity_v1 left, stable_identity_v1 right) noexcept {
    return left.low == right.low && left.high == right.high;
}

inline export_status_v1 export_atom_profile_manifest_v1(
    const atom_profile_record_v1 *records, std::uint64_t record_count,
    const profile_export_sink_v1 &sink) noexcept {
    if (records == nullptr || record_count == 0u || sink.emit == nullptr) {
        return {export_code_v1::invalid_argument};
    }
    for (std::uint64_t atom_index = 0u; atom_index < record_count;
         ++atom_index) {
        const auto &record = records[atom_index];
        if (!valid_identity_v1(record.atom_identity) ||
            !valid_identity_v1(record.atom_species_identity)) {
            return {export_code_v1::invalid_identity, atom_index};
        }
        if (record.structure_epoch == 0u || record.mechanism_count == 0u ||
            record.mechanisms == nullptr) {
            return {export_code_v1::invalid_argument, atom_index};
        }
        for (std::uint64_t prior = 0u; prior < atom_index; ++prior) {
            if (same_identity_v1(
                    records[prior].atom_identity, record.atom_identity)) {
                return {export_code_v1::duplicate_atom, atom_index};
            }
        }
        for (std::uint64_t mechanism_index = 0u;
             mechanism_index < record.mechanism_count; ++mechanism_index) {
            const auto &mechanism = record.mechanisms[mechanism_index];
            if (!valid_identity_v1(mechanism.mechanism_identity) ||
                mechanism.candidate_id == 0u || mechanism.kernel_id == 0u ||
                mechanism.useful_interactions == 0u ||
                mechanism.launch_count == 0u) {
                return {export_code_v1::invalid_measurement, atom_index,
                    mechanism_index};
            }
        }
        if (!sink.emit(sink.context, record)) {
            return {export_code_v1::sink_failed, atom_index};
        }
    }
    return {};
}

static_assert(std::is_trivially_copyable_v<mechanism_manifest_v1>);
static_assert(std::is_trivially_copyable_v<atom_profile_record_v1>);
static_assert(std::is_trivially_copyable_v<profile_export_sink_v1>);

}  // namespace cellerator::profiling::joint_compiler
