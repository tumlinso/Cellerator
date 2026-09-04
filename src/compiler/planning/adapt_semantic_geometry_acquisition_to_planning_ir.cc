#include <Cellerator/compiler/planning/adapt_semantic_geometry_acquisition_to_planning_ir_v1.hh>

#include <cstring>
#include <type_traits>

namespace Cellerator::compiler::planning {
namespace {

constexpr std::array<std::byte, 4> csg1_magic{
    std::byte{'C'}, std::byte{'S'}, std::byte{'G'}, std::byte{'1'}};

enum class csg1_record_kind : std::uint8_t { request = 1u, result = 2u };

struct csg1_header {
    std::array<std::byte, 4> magic{};
    std::uint8_t kind = 0u;
    std::array<std::uint8_t, 3> reserved{};
    std::uint32_t payload_bytes = 0u;
};

constexpr std::uint32_t all_compatibility = compatible_semantics_v1 |
    compatible_profile_v1 | compatible_target_v1 | exact_logical_coverage_v1;

constexpr bool valid_kind(geometry_acquisition_kind_v1 kind) noexcept {
    return kind >= geometry_acquisition_kind_v1::compile_now &&
        kind <= geometry_acquisition_kind_v1::conventional_fallback;
}

template<typename Record>
std::vector<std::byte> encode(csg1_record_kind kind, const Record& record) {
    static_assert(std::is_trivially_copyable_v<Record>);
    csg1_header header{csg1_magic, static_cast<std::uint8_t>(kind), {},
                       static_cast<std::uint32_t>(sizeof(Record))};
    std::vector<std::byte> bytes(sizeof(header) + sizeof(record));
    std::memcpy(bytes.data(), &header, sizeof(header));
    std::memcpy(bytes.data() + sizeof(header), &record, sizeof(record));
    return bytes;
}

template<typename Record>
geometry_acquisition_validation_code_v1 decode(
    csg1_record_kind expected,
    const std::byte* data,
    std::size_t bytes,
    Record* record) noexcept {
    if (data == nullptr || record == nullptr ||
        bytes != sizeof(csg1_header) + sizeof(Record)) {
        return geometry_acquisition_validation_code_v1::malformed_csg1;
    }
    csg1_header header{};
    std::memcpy(&header, data, sizeof(header));
    if (header.magic != csg1_magic ||
        header.kind != static_cast<std::uint8_t>(expected) ||
        header.payload_bytes != sizeof(Record)) {
        return geometry_acquisition_validation_code_v1::malformed_csg1;
    }
    std::memcpy(record, data + sizeof(header), sizeof(Record));
    return geometry_acquisition_validation_code_v1::ok;
}

}  // namespace

geometry_acquisition_validation_code_v1 validate_geometry_acquisition_request_v1(
    const geometry_acquisition_request_v1& request) noexcept {
    if (request.schema_version != semantic_geometry_acquisition_schema_v1) {
        return geometry_acquisition_validation_code_v1::unsupported_schema;
    }
    if (request.record_bytes != sizeof(request)) {
        return geometry_acquisition_validation_code_v1::invalid_record_bytes;
    }
    if (!valid_kind(request.kind)) {
        return geometry_acquisition_validation_code_v1::invalid_kind;
    }
    if (!request.request_identity.valid() ||
        !request.semantic_problem_identity.valid() ||
        !request.profile_identity.valid() || !request.target_identity.valid()) {
        return geometry_acquisition_validation_code_v1::invalid_identity;
    }
    if (request.required_compatibility == 0u ||
        (request.required_compatibility & ~all_compatibility) != 0u) {
        return geometry_acquisition_validation_code_v1::invalid_compatibility;
    }
    if (request.kind == geometry_acquisition_kind_v1::external_exact_cover &&
        (request.required_compatibility & exact_logical_coverage_v1) == 0u) {
        return geometry_acquisition_validation_code_v1::exact_cover_not_required;
    }
    if ((request.kind == geometry_acquisition_kind_v1::precompiled_semantic_geometry ||
         request.kind == geometry_acquisition_kind_v1::external_exact_cover) &&
        !request.supplied_geometry_identity.valid()) {
        return geometry_acquisition_validation_code_v1::supplied_geometry_missing;
    }
    return geometry_acquisition_validation_code_v1::ok;
}

geometry_acquisition_validation_code_v1 validate_geometry_acquisition_result_v1(
    const geometry_acquisition_request_v1& request,
    const geometry_acquisition_result_v1& result) noexcept {
    const auto request_status = validate_geometry_acquisition_request_v1(request);
    if (request_status != geometry_acquisition_validation_code_v1::ok) {
        return request_status;
    }
    if (result.schema_version != semantic_geometry_acquisition_schema_v1) {
        return geometry_acquisition_validation_code_v1::unsupported_schema;
    }
    if (result.record_bytes != sizeof(result)) {
        return geometry_acquisition_validation_code_v1::invalid_record_bytes;
    }
    if (!valid_kind(result.kind)) {
        return geometry_acquisition_validation_code_v1::invalid_kind;
    }
    if (result.kind != request.kind ||
        result.request_identity.low != request.request_identity.low ||
        result.request_identity.high != request.request_identity.high) {
        return geometry_acquisition_validation_code_v1::result_mismatch;
    }
    if (result.status == geometry_acquisition_status_v1::acquired) {
        if (!result.semantic_geometry_identity.valid() ||
            !result.provider_identity.valid()) {
            return geometry_acquisition_validation_code_v1::invalid_identity;
        }
        if ((result.satisfied_compatibility & request.required_compatibility) !=
            request.required_compatibility) {
            return geometry_acquisition_validation_code_v1::invalid_compatibility;
        }
    }
    return geometry_acquisition_validation_code_v1::ok;
}

std::vector<std::byte> encode_csg1_request_v1(
    const geometry_acquisition_request_v1& request) {
    return encode(csg1_record_kind::request, request);
}

std::vector<std::byte> encode_csg1_result_v1(
    const geometry_acquisition_result_v1& result) {
    return encode(csg1_record_kind::result, result);
}

geometry_acquisition_validation_code_v1 decode_csg1_request_v1(
    const std::byte* data,
    std::size_t bytes,
    geometry_acquisition_request_v1* request) noexcept {
    const auto status = decode(csg1_record_kind::request, data, bytes, request);
    return status == geometry_acquisition_validation_code_v1::ok
        ? validate_geometry_acquisition_request_v1(*request) : status;
}

geometry_acquisition_validation_code_v1 decode_csg1_result_v1(
    const std::byte* data,
    std::size_t bytes,
    geometry_acquisition_result_v1* result) noexcept {
    return decode(csg1_record_kind::result, data, bytes, result);
}

}  // namespace Cellerator::compiler::planning
