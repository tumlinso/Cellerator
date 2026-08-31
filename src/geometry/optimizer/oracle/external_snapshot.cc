#include "Cellerator/geometry/optimizer/oracle/external_snapshot.h"

#include <cstring>
#include <limits>
#include <type_traits>

namespace cellerator::geometry::optimizer::oracle {
namespace {

constexpr std::uint8_t snapshot_magic[8] = {'C', 'E', 'O', 'R', 'C', 'L', '1', 0};
constexpr std::uint32_t snapshot_version = 1;
constexpr std::uint32_t little_endian_tag = 0x01020304U;
constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

struct alignas(8) snapshot_header {
    std::uint8_t magic[8];
    std::uint32_t version;
    std::uint32_t endian_tag;
    std::uint64_t header_bytes;
    std::uint64_t total_bytes;
    std::uint64_t contribution_count;
    std::uint64_t candidate_count;
    std::uint64_t coverage_offsets_offset;
    std::uint64_t coverage_offsets_count;
    std::uint64_t coverage_indices_offset;
    std::uint64_t coverage_indices_count;
    std::uint64_t candidate_costs_offset;
    std::uint64_t candidate_costs_count;
    std::uint64_t residual_costs_offset;
    std::uint64_t residual_costs_count;
    std::int64_t fixed_cost;
    std::uint64_t payload_fingerprint;
};

static_assert(sizeof(snapshot_header) % 8 == 0);
static_assert(std::is_trivially_copyable_v<snapshot_header>);

bool host_is_little_endian() noexcept {
    const std::uint32_t value = little_endian_tag;
    return *reinterpret_cast<const std::uint8_t*>(&value) == 4;
}

bool checked_add(std::uint64_t lhs, std::uint64_t rhs, std::uint64_t* out) noexcept {
    if (lhs > std::numeric_limits<std::uint64_t>::max() - rhs) {
        return false;
    }
    *out = lhs + rhs;
    return true;
}

bool checked_multiply(
        std::uint64_t lhs,
        std::uint64_t rhs,
        std::uint64_t* out) noexcept {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

bool append_section(
        std::uint64_t count,
        std::uint64_t element_bytes,
        std::uint64_t* cursor,
        std::uint64_t* offset) noexcept {
    *offset = *cursor;
    std::uint64_t section_bytes = 0;
    return checked_multiply(count, element_bytes, &section_bytes) &&
           checked_add(*cursor, section_bytes, cursor);
}

std::uint64_t fingerprint_bytes(
        const std::byte* bytes,
        std::uint64_t byte_count,
        std::uint64_t hash) noexcept {
    for (std::uint64_t index = 0; index < byte_count; ++index) {
        hash ^= static_cast<std::uint8_t>(bytes[index]);
        hash *= fnv_prime;
    }
    return hash;
}

std::uint64_t snapshot_fingerprint(
        const std::byte* bytes,
        std::uint64_t total_bytes) noexcept {
    constexpr std::uint64_t fingerprint_offset =
            offsetof(snapshot_header, payload_fingerprint);
    std::uint64_t hash = fingerprint_bytes(bytes, fingerprint_offset, fnv_offset);
    return fingerprint_bytes(
            bytes + sizeof(snapshot_header),
            total_bytes - sizeof(snapshot_header),
            hash);
}

bool section_is_valid(
        std::uint64_t offset,
        std::uint64_t count,
        std::uint64_t element_bytes,
        std::uint64_t total_bytes) noexcept {
    if (offset % 8 != 0) {
        return false;
    }
    std::uint64_t section_bytes = 0;
    std::uint64_t end = 0;
    return checked_multiply(count, element_bytes, &section_bytes) &&
           checked_add(offset, section_bytes, &end) &&
           end <= total_bytes;
}

}  // namespace

oracle_snapshot_requirements query_oracle_snapshot_requirements(
        const exact_oracle_problem_view& problem) noexcept {
    oracle_snapshot_requirements requirements{};
    if ((problem.candidate_count != 0 && problem.coverage_offsets == nullptr) ||
        (problem.contribution_count != 0 && problem.residual_costs == nullptr)) {
        requirements.status = oracle_snapshot_status::invalid_argument;
        return requirements;
    }
    const std::uint64_t coverage_count =
            problem.candidate_count == 0 ? 0 : problem.coverage_offsets[problem.candidate_count];
    if ((coverage_count != 0 && problem.coverage_indices == nullptr) ||
        (problem.candidate_count != 0 && problem.candidate_costs == nullptr)) {
        requirements.status = oracle_snapshot_status::invalid_argument;
        return requirements;
    }
    if (problem.candidate_count != 0 && problem.coverage_offsets[0] != 0) {
        requirements.status = oracle_snapshot_status::invalid_problem;
        return requirements;
    }
    for (std::uint32_t candidate = 0; candidate < problem.candidate_count; ++candidate) {
        const std::uint64_t begin = problem.coverage_offsets[candidate];
        const std::uint64_t end = problem.coverage_offsets[candidate + 1];
        if (end < begin || end > coverage_count) {
            requirements.status = oracle_snapshot_status::invalid_problem;
            return requirements;
        }
        std::uint64_t previous = 0;
        for (std::uint64_t offset = begin; offset < end; ++offset) {
            const std::uint64_t contribution = problem.coverage_indices[offset];
            if (contribution >= problem.contribution_count ||
                (offset != begin && contribution <= previous)) {
                requirements.status = oracle_snapshot_status::invalid_problem;
                return requirements;
            }
            previous = contribution;
        }
    }
    std::uint64_t cursor = sizeof(snapshot_header);
    std::uint64_t ignored = 0;
    if (!append_section(
                static_cast<std::uint64_t>(problem.candidate_count) + 1,
                sizeof(std::uint64_t), &cursor, &ignored) ||
        !append_section(coverage_count, sizeof(std::uint64_t), &cursor, &ignored) ||
        !append_section(problem.candidate_count, sizeof(std::int64_t), &cursor, &ignored) ||
        !append_section(problem.contribution_count, sizeof(std::int64_t), &cursor, &ignored)) {
        requirements.status = oracle_snapshot_status::arithmetic_overflow;
        return requirements;
    }
    requirements.status = oracle_snapshot_status::success;
    requirements.bytes = cursor;
    return requirements;
}

oracle_snapshot_status write_oracle_snapshot(
        const exact_oracle_problem_view& problem,
        const mutable_oracle_snapshot_view& destination,
        std::uint64_t* bytes_written) noexcept {
    if (bytes_written == nullptr || !host_is_little_endian()) {
        return bytes_written == nullptr
                ? oracle_snapshot_status::invalid_argument
                : oracle_snapshot_status::unsupported_encoding;
    }
    *bytes_written = 0;
    const auto requirements = query_oracle_snapshot_requirements(problem);
    if (requirements.status != oracle_snapshot_status::success) {
        return requirements.status;
    }
    if (requirements.bytes > std::numeric_limits<std::size_t>::max() ||
        destination.byte_capacity < requirements.bytes ||
        (requirements.bytes != 0 && destination.bytes == nullptr) ||
        reinterpret_cast<std::uintptr_t>(destination.bytes) % requirements.alignment != 0) {
        return oracle_snapshot_status::capacity_exceeded;
    }

    snapshot_header header{};
    std::memcpy(header.magic, snapshot_magic, sizeof(snapshot_magic));
    header.version = snapshot_version;
    header.endian_tag = little_endian_tag;
    header.header_bytes = sizeof(snapshot_header);
    header.total_bytes = requirements.bytes;
    header.contribution_count = problem.contribution_count;
    header.candidate_count = problem.candidate_count;
    header.coverage_offsets_count = static_cast<std::uint64_t>(problem.candidate_count) + 1;
    header.coverage_indices_count =
            problem.candidate_count == 0 ? 0 : problem.coverage_offsets[problem.candidate_count];
    header.candidate_costs_count = problem.candidate_count;
    header.residual_costs_count = problem.contribution_count;
    header.fixed_cost = problem.fixed_cost;
    std::uint64_t cursor = sizeof(snapshot_header);
    append_section(header.coverage_offsets_count, sizeof(std::uint64_t),
                   &cursor, &header.coverage_offsets_offset);
    append_section(header.coverage_indices_count, sizeof(std::uint64_t),
                   &cursor, &header.coverage_indices_offset);
    append_section(header.candidate_costs_count, sizeof(std::int64_t),
                   &cursor, &header.candidate_costs_offset);
    append_section(header.residual_costs_count, sizeof(std::int64_t),
                   &cursor, &header.residual_costs_offset);

    std::memcpy(destination.bytes, &header, sizeof(header));
    if (header.coverage_offsets_count != 0) {
        if (problem.candidate_count == 0) {
            const std::uint64_t zero = 0;
            std::memcpy(destination.bytes + header.coverage_offsets_offset,
                        &zero, sizeof(zero));
        } else {
            std::memcpy(destination.bytes + header.coverage_offsets_offset,
                        problem.coverage_offsets,
                        header.coverage_offsets_count * sizeof(std::uint64_t));
        }
    }
    if (header.coverage_indices_count != 0) {
        std::memcpy(destination.bytes + header.coverage_indices_offset,
                    problem.coverage_indices,
                    header.coverage_indices_count * sizeof(std::uint64_t));
    }
    if (header.candidate_costs_count != 0) {
        std::memcpy(destination.bytes + header.candidate_costs_offset,
                    problem.candidate_costs,
                    header.candidate_costs_count * sizeof(std::int64_t));
    }
    if (header.residual_costs_count != 0) {
        std::memcpy(destination.bytes + header.residual_costs_offset,
                    problem.residual_costs,
                    header.residual_costs_count * sizeof(std::int64_t));
    }
    header.payload_fingerprint = snapshot_fingerprint(
            destination.bytes, requirements.bytes);
    std::memcpy(destination.bytes, &header, sizeof(header));
    *bytes_written = requirements.bytes;
    return oracle_snapshot_status::success;
}

parsed_oracle_snapshot parse_oracle_snapshot(
        const oracle_snapshot_view& snapshot) noexcept {
    parsed_oracle_snapshot parsed{};
    if (snapshot.bytes == nullptr || snapshot.byte_count < sizeof(snapshot_header)) {
        parsed.status = oracle_snapshot_status::invalid_argument;
        return parsed;
    }
    if (!host_is_little_endian()) {
        parsed.status = oracle_snapshot_status::unsupported_encoding;
        return parsed;
    }
    snapshot_header header{};
    std::memcpy(&header, snapshot.bytes, sizeof(header));
    if (std::memcmp(header.magic, snapshot_magic, sizeof(snapshot_magic)) != 0 ||
        header.version != snapshot_version || header.endian_tag != little_endian_tag ||
        header.header_bytes != sizeof(snapshot_header)) {
        parsed.status = oracle_snapshot_status::unsupported_encoding;
        return parsed;
    }
    std::uint64_t expected_cursor = sizeof(snapshot_header);
    std::uint64_t expected_offsets_offset = 0;
    std::uint64_t expected_indices_offset = 0;
    std::uint64_t expected_candidate_costs_offset = 0;
    std::uint64_t expected_residual_costs_offset = 0;
    const bool expected_layout_valid =
            append_section(header.coverage_offsets_count, sizeof(std::uint64_t),
                           &expected_cursor, &expected_offsets_offset) &&
            append_section(header.coverage_indices_count, sizeof(std::uint64_t),
                           &expected_cursor, &expected_indices_offset) &&
            append_section(header.candidate_costs_count, sizeof(std::int64_t),
                           &expected_cursor, &expected_candidate_costs_offset) &&
            append_section(header.residual_costs_count, sizeof(std::int64_t),
                           &expected_cursor, &expected_residual_costs_offset);
    if (header.total_bytes > snapshot.byte_count ||
        header.total_bytes > std::numeric_limits<std::size_t>::max() ||
        header.candidate_count > std::numeric_limits<std::uint32_t>::max() ||
        header.coverage_offsets_count != header.candidate_count + 1 ||
        header.candidate_costs_count != header.candidate_count ||
        header.residual_costs_count != header.contribution_count ||
        !expected_layout_valid || expected_cursor != header.total_bytes ||
        expected_offsets_offset != header.coverage_offsets_offset ||
        expected_indices_offset != header.coverage_indices_offset ||
        expected_candidate_costs_offset != header.candidate_costs_offset ||
        expected_residual_costs_offset != header.residual_costs_offset ||
        !section_is_valid(header.coverage_offsets_offset, header.coverage_offsets_count,
                          sizeof(std::uint64_t), header.total_bytes) ||
        !section_is_valid(header.coverage_indices_offset, header.coverage_indices_count,
                          sizeof(std::uint64_t), header.total_bytes) ||
        !section_is_valid(header.candidate_costs_offset, header.candidate_costs_count,
                          sizeof(std::int64_t), header.total_bytes) ||
        !section_is_valid(header.residual_costs_offset, header.residual_costs_count,
                          sizeof(std::int64_t), header.total_bytes)) {
        parsed.status = oracle_snapshot_status::corrupt_snapshot;
        return parsed;
    }
    const auto actual_fingerprint = snapshot_fingerprint(
            snapshot.bytes, header.total_bytes);
    if (actual_fingerprint != header.payload_fingerprint) {
        parsed.status = oracle_snapshot_status::corrupt_snapshot;
        return parsed;
    }
    parsed.problem.contribution_count = header.contribution_count;
    parsed.problem.candidate_count = static_cast<std::uint32_t>(header.candidate_count);
    parsed.problem.coverage_offsets = reinterpret_cast<const std::uint64_t*>(
            snapshot.bytes + header.coverage_offsets_offset);
    parsed.problem.coverage_indices = reinterpret_cast<const std::uint64_t*>(
            snapshot.bytes + header.coverage_indices_offset);
    parsed.problem.candidate_costs = reinterpret_cast<const std::int64_t*>(
            snapshot.bytes + header.candidate_costs_offset);
    parsed.problem.residual_costs = reinterpret_cast<const std::int64_t*>(
            snapshot.bytes + header.residual_costs_offset);
    parsed.problem.fixed_cost = header.fixed_cost;
    const auto requirements = query_oracle_snapshot_requirements(parsed.problem);
    if (requirements.status != oracle_snapshot_status::success ||
        requirements.bytes != header.total_bytes) {
        parsed.problem = {};
        parsed.status = oracle_snapshot_status::corrupt_snapshot;
        return parsed;
    }
    parsed.encoded_bytes = header.total_bytes;
    parsed.payload_fingerprint = header.payload_fingerprint;
    parsed.status = oracle_snapshot_status::success;
    return parsed;
}

dummy_alternative_result solve_dummy_alternative(
        const exact_oracle_problem_view& problem,
        std::uint8_t* selection,
        std::uint32_t selection_capacity,
        std::uint32_t* contribution_owners,
        std::uint64_t contribution_owner_capacity) noexcept {
    dummy_alternative_result result{};
    if (selection_capacity < problem.candidate_count ||
        (problem.candidate_count != 0 && selection == nullptr)) {
        result.status = exact_oracle_status::insufficient_workspace;
        return result;
    }
    if (problem.candidate_count != 0) {
        std::memset(selection, 0, problem.candidate_count);
    }
    const auto evaluation = evaluate_exact_oracle_selection(
            problem, selection, problem.candidate_count,
            contribution_owners, contribution_owner_capacity);
    result.status = evaluation.status;
    result.objective = evaluation.objective;
    result.selected_candidate_count = evaluation.selected_candidate_count;
    result.admissible = evaluation.admissible;
    return result;
}

}  // namespace cellerator::geometry::optimizer::oracle
