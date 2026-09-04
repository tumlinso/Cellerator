#include <Cellerator/compiler/discovery/import_the_overlapping_evidence_atlas_core_v1.hh>

#include <algorithm>
#include <limits>

namespace Cellerator::compiler::discovery {
namespace {

constexpr std::uint64_t atlas_magic = 0x31414c5441454343ULL;
constexpr std::size_t header_bytes = 56;
constexpr std::size_t record_bytes = 136;

void set_status(evidence_atlas_status_v1* status, evidence_atlas_status_v1 value) {
    if (status != nullptr) *status = value;
}

void put_u64(std::vector<std::uint8_t>& bytes, std::uint64_t value) {
    for (unsigned index = 0; index < 8; ++index) {
        bytes.push_back(static_cast<std::uint8_t>(value & 0xffu));
        value >>= 8u;
    }
}

std::uint64_t get_u64(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    std::uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index)
        value |= static_cast<std::uint64_t>(bytes[offset + index]) << (8u * index);
    return value;
}

void put_identity(std::vector<std::uint8_t>& bytes, persistent_atom_identity_v1 value) {
    put_u64(bytes, value.producer_namespace);
    put_u64(bytes, value.local_identity);
}

persistent_atom_identity_v1 get_identity(
    const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    return {get_u64(bytes, offset), get_u64(bytes, offset + 8)};
}

std::uint64_t checksum(const std::vector<std::uint8_t>& bytes,
                       std::size_t end) noexcept {
    std::uint64_t value = 14695981039346656037ULL;
    for (std::size_t index = 0; index < end; ++index)
        value = (value ^ bytes[index]) * 1099511628211ULL;
    return value;
}

bool equal_record(const proposal_evidence_record_v1& left,
                  const proposal_evidence_record_v1& right) noexcept {
    return left.evidence_identity == right.evidence_identity &&
        left.subject_atom_identity == right.subject_atom_identity &&
        left.provenance_identity == right.provenance_identity &&
        left.observation_generation == right.observation_generation &&
        left.approximate_members == right.approximate_members &&
        left.confidence_numerator == right.confidence_numerator &&
        left.confidence_denominator == right.confidence_denominator &&
        left.stable_resamples == right.stable_resamples &&
        left.total_resamples == right.total_resamples &&
        left.exact_visited == right.exact_visited &&
        left.exact_assigned == right.exact_assigned &&
        left.negative_reason == right.negative_reason &&
        left.exact_rescan == right.exact_rescan;
}

}  // namespace

evidence_atlas_status_v1 validate_overlapping_evidence_atlas_v1(
    const overlapping_evidence_atlas_v1& atlas) noexcept {
    if (!valid_persistent_atom_identity_v1(atlas.atlas_identity))
        return evidence_atlas_status_v1::invalid_atlas_identity;
    if (atlas.generation == 0) return evidence_atlas_status_v1::missing_generation;
    if (atlas.proposals.empty()) return evidence_atlas_status_v1::empty_atlas;
    for (std::size_t index = 0; index < atlas.proposals.size(); ++index) {
        const auto& record = atlas.proposals[index];
        if (!valid_persistent_atom_identity_v1(record.evidence_identity) ||
            !valid_persistent_atom_identity_v1(record.subject_atom_identity) ||
            !valid_persistent_atom_identity_v1(record.provenance_identity) ||
            record.observation_generation == 0)
            return evidence_atlas_status_v1::invalid_record_identity;
        if (index != 0 && !persistent_atom_identity_less_v1(
                atlas.proposals[index - 1].evidence_identity,
                record.evidence_identity))
            return evidence_atlas_status_v1::duplicate_record;
        for (std::size_t member = 0; member < record.approximate_members.size(); ++member) {
            if (!valid_persistent_atom_identity_v1(record.approximate_members[member]))
                return evidence_atlas_status_v1::invalid_member;
            if (member != 0 && !persistent_atom_identity_less_v1(
                    record.approximate_members[member - 1],
                    record.approximate_members[member]))
                return evidence_atlas_status_v1::unordered_or_duplicate_member;
        }
        if (record.confidence_denominator == 0 ||
            record.confidence_numerator > record.confidence_denominator)
            return evidence_atlas_status_v1::invalid_confidence;
        if (record.total_resamples == 0 ||
            record.stable_resamples > record.total_resamples)
            return evidence_atlas_status_v1::invalid_stability;
        if (record.exact_assigned > record.exact_visited ||
            (record.exact_rescan == exact_rescan_status_v1::complete &&
             record.exact_visited == 0))
            return evidence_atlas_status_v1::invalid_exact_rescan;
        const auto reason = static_cast<std::uint8_t>(record.negative_reason);
        if (reason > 6) return evidence_atlas_status_v1::invalid_negative_reason;
    }
    return evidence_atlas_status_v1::success;
}

std::optional<std::vector<std::uint8_t>> serialize_overlapping_evidence_atlas_v1(
    const overlapping_evidence_atlas_v1& atlas,
    evidence_atlas_status_v1* status) noexcept {
    const auto validation = validate_overlapping_evidence_atlas_v1(atlas);
    if (validation != evidence_atlas_status_v1::success) {
        set_status(status, validation);
        return std::nullopt;
    }
    std::uint64_t member_count = 0;
    for (const auto& record : atlas.proposals) member_count += record.approximate_members.size();
    if (atlas.proposals.size() > (std::numeric_limits<std::size_t>::max() - header_bytes) /
                                     record_bytes ||
        member_count > (std::numeric_limits<std::size_t>::max() - header_bytes -
                        atlas.proposals.size() * record_bytes) / 16) {
        set_status(status, evidence_atlas_status_v1::invalid_image);
        return std::nullopt;
    }
    std::vector<std::uint8_t> bytes;
    bytes.reserve(header_bytes + atlas.proposals.size() * record_bytes + member_count * 16);
    put_u64(bytes, atlas_magic);
    put_u64(bytes, 1);
    put_identity(bytes, atlas.atlas_identity);
    put_u64(bytes, atlas.generation);
    put_u64(bytes, atlas.proposals.size());
    put_u64(bytes, member_count);
    std::uint64_t member_offset = 0;
    for (const auto& record : atlas.proposals) {
        put_identity(bytes, record.evidence_identity);
        put_identity(bytes, record.subject_atom_identity);
        put_identity(bytes, record.provenance_identity);
        put_u64(bytes, record.observation_generation);
        put_u64(bytes, member_offset);
        put_u64(bytes, record.approximate_members.size());
        put_u64(bytes, record.confidence_numerator);
        put_u64(bytes, record.confidence_denominator);
        put_u64(bytes, record.stable_resamples);
        put_u64(bytes, record.total_resamples);
        put_u64(bytes, record.exact_visited);
        put_u64(bytes, record.exact_assigned);
        put_u64(bytes, static_cast<std::uint64_t>(record.negative_reason));
        put_u64(bytes, static_cast<std::uint64_t>(record.exact_rescan));
        member_offset += record.approximate_members.size();
    }
    for (const auto& record : atlas.proposals)
        for (const auto member : record.approximate_members) put_identity(bytes, member);
    put_u64(bytes, checksum(bytes, bytes.size()));
    set_status(status, evidence_atlas_status_v1::success);
    return bytes;
}

std::optional<overlapping_evidence_atlas_v1> deserialize_overlapping_evidence_atlas_v1(
    const std::vector<std::uint8_t>& image,
    evidence_atlas_status_v1* status) noexcept {
    if (image.size() < header_bytes + 8 || get_u64(image, 0) != atlas_magic ||
        get_u64(image, 8) != 1) {
        set_status(status, evidence_atlas_status_v1::invalid_image);
        return std::nullopt;
    }
    const auto record_count = get_u64(image, 40);
    const auto member_count = get_u64(image, 48);
    if (record_count > (std::numeric_limits<std::size_t>::max() - header_bytes - 8) /
                           record_bytes ||
        header_bytes + record_count * record_bytes > image.size() - 8 ||
        member_count != (image.size() - 8 - header_bytes - record_count * record_bytes) / 16 ||
        header_bytes + record_count * record_bytes + member_count * 16 + 8 != image.size()) {
        set_status(status, evidence_atlas_status_v1::invalid_image);
        return std::nullopt;
    }
    if (get_u64(image, image.size() - 8) != checksum(image, image.size() - 8)) {
        set_status(status, evidence_atlas_status_v1::checksum_mismatch);
        return std::nullopt;
    }
    overlapping_evidence_atlas_v1 atlas;
    atlas.atlas_identity = get_identity(image, 16);
    atlas.generation = get_u64(image, 32);
    try { atlas.proposals.reserve(record_count); } catch (...) {
        set_status(status, evidence_atlas_status_v1::invalid_image);
        return std::nullopt;
    }
    const auto members_begin = header_bytes + record_count * record_bytes;
    for (std::uint64_t index = 0; index < record_count; ++index) {
        const auto offset = header_bytes + index * record_bytes;
        proposal_evidence_record_v1 record;
        record.evidence_identity = get_identity(image, offset);
        record.subject_atom_identity = get_identity(image, offset + 16);
        record.provenance_identity = get_identity(image, offset + 32);
        record.observation_generation = get_u64(image, offset + 48);
        const auto member_offset = get_u64(image, offset + 56);
        const auto count = get_u64(image, offset + 64);
        if (member_offset > member_count || count > member_count - member_offset) {
            set_status(status, evidence_atlas_status_v1::invalid_image);
            return std::nullopt;
        }
        record.confidence_numerator = get_u64(image, offset + 72);
        record.confidence_denominator = get_u64(image, offset + 80);
        record.stable_resamples = get_u64(image, offset + 88);
        record.total_resamples = get_u64(image, offset + 96);
        record.exact_visited = get_u64(image, offset + 104);
        record.exact_assigned = get_u64(image, offset + 112);
        record.negative_reason = static_cast<negative_evidence_reason_v1>(
            get_u64(image, offset + 120));
        record.exact_rescan = static_cast<exact_rescan_status_v1>(
            get_u64(image, offset + 128));
        try {
            for (std::uint64_t member = 0; member < count; ++member)
                record.approximate_members.push_back(get_identity(
                    image, members_begin + (member_offset + member) * 16));
            atlas.proposals.push_back(std::move(record));
        } catch (...) {
            set_status(status, evidence_atlas_status_v1::invalid_image);
            return std::nullopt;
        }
    }
    const auto validation = validate_overlapping_evidence_atlas_v1(atlas);
    if (validation != evidence_atlas_status_v1::success) {
        set_status(status, validation);
        return std::nullopt;
    }
    set_status(status, evidence_atlas_status_v1::success);
    return atlas;
}

bool equivalent_evidence_atlas_v1(const overlapping_evidence_atlas_v1& left,
                                  const overlapping_evidence_atlas_v1& right) noexcept {
    return left.atlas_identity == right.atlas_identity &&
        left.generation == right.generation &&
        left.proposals.size() == right.proposals.size() &&
        std::equal(left.proposals.begin(), left.proposals.end(),
                   right.proposals.begin(), equal_record);
}

}  // namespace Cellerator::compiler::discovery
