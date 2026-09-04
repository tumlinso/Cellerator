#include <Cellerator/compiler/planning/implement_custom_candidate_registration_v1.hh>

#include <algorithm>
#include <limits>

namespace Cellerator::compiler::planning {
namespace {

constexpr std::uint8_t magic[] = {'C', 'E', 'C', 'C', 1u};
constexpr char hex[] = "0123456789abcdef";

void append_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
    for (unsigned shift = 0u; shift < 32u; shift += 8u)
        out.push_back(static_cast<std::uint8_t>(value >> shift));
}

void append_u64(std::vector<std::uint8_t>& out, std::uint64_t value) {
    for (unsigned shift = 0u; shift < 64u; shift += 8u)
        out.push_back(static_cast<std::uint8_t>(value >> shift));
}

void append_bytes(std::vector<std::uint8_t>& out, const std::uint8_t* data,
                  std::size_t size) {
    append_u64(out, size);
    out.insert(out.end(), data, data + size);
}

void append_string(std::vector<std::uint8_t>& out, const std::string& value) {
    append_bytes(out, reinterpret_cast<const std::uint8_t*>(value.data()), value.size());
}

struct reader {
    const std::vector<std::uint8_t>& bytes;
    std::size_t offset = 0u;

    bool u32(std::uint32_t* value) {
        if (bytes.size() - offset < 4u) return false;
        *value = 0u;
        for (unsigned shift = 0u; shift < 32u; shift += 8u)
            *value |= static_cast<std::uint32_t>(bytes[offset++]) << shift;
        return true;
    }
    bool u64(std::uint64_t* value) {
        if (bytes.size() - offset < 8u) return false;
        *value = 0u;
        for (unsigned shift = 0u; shift < 64u; shift += 8u)
            *value |= static_cast<std::uint64_t>(bytes[offset++]) << shift;
        return true;
    }
    bool blob(std::vector<std::uint8_t>* value) {
        std::uint64_t size = 0u;
        if (!u64(&size) || size > bytes.size() - offset) return false;
        value->assign(bytes.begin() + offset, bytes.begin() + offset + size);
        offset += static_cast<std::size_t>(size);
        return true;
    }
    bool string(std::string* value) {
        std::vector<std::uint8_t> data;
        if (!blob(&data)) return false;
        value->assign(data.begin(), data.end());
        return true;
    }
};

custom_candidate_registration_code_v1 validate(
    const custom_candidate_registration_v1& candidate) noexcept {
    if (candidate.candidate_identity == 0u || candidate.provider_identity == 0u ||
        candidate.operation_identity == 0u)
        return custom_candidate_registration_code_v1::invalid_identity;
    if (candidate.stable_name.empty())
        return custom_candidate_registration_code_v1::invalid_name;
    if (candidate.origin < custom_candidate_origin_v1::source ||
        candidate.origin > custom_candidate_origin_v1::migrated_provider)
        return custom_candidate_registration_code_v1::invalid_origin;
    if (candidate.source_locator.empty())
        return custom_candidate_registration_code_v1::missing_source_locator;
    if (candidate.origin == custom_candidate_origin_v1::external_library &&
        candidate.entry_symbol.empty())
        return custom_candidate_registration_code_v1::missing_entry_symbol;
    const auto missing = candidate.required_protocols & ~candidate.provided_protocols;
    if (missing != 0u && candidate.missing_behavior !=
        missing_protocol_behavior_v1::opaque_passthrough)
        return custom_candidate_registration_code_v1::incomplete_protocol;
    return custom_candidate_registration_code_v1::ok;
}

bool same(const custom_candidate_registration_v1& lhs,
          const custom_candidate_registration_v1& rhs) noexcept {
    return lhs.candidate_identity == rhs.candidate_identity &&
        lhs.provider_identity == rhs.provider_identity &&
        lhs.operation_identity == rhs.operation_identity && lhs.origin == rhs.origin &&
        lhs.provided_protocols == rhs.provided_protocols &&
        lhs.required_protocols == rhs.required_protocols &&
        lhs.missing_behavior == rhs.missing_behavior && lhs.stable_name == rhs.stable_name &&
        lhs.source_locator == rhs.source_locator && lhs.entry_symbol == rhs.entry_symbol &&
        lhs.opaque_payload == rhs.opaque_payload;
}

int nibble(char value) noexcept {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    return -1;
}

}  // namespace

custom_candidate_registration_code_v1 register_custom_candidate_v1(
    custom_candidate_registry_v1* registry,
    custom_candidate_registration_v1 candidate) {
    if (registry == nullptr) return custom_candidate_registration_code_v1::malformed_ir;
    const auto code = validate(candidate);
    if (code != custom_candidate_registration_code_v1::ok) return code;
    if (std::any_of(registry->candidates.begin(), registry->candidates.end(),
        [&](const auto& item) { return item.candidate_identity == candidate.candidate_identity; }))
        return custom_candidate_registration_code_v1::duplicate_candidate;
    registry->candidates.push_back(std::move(candidate));
    return custom_candidate_registration_code_v1::ok;
}

std::vector<std::uint8_t> write_custom_candidate_binary_ir_v1(
    const custom_candidate_registry_v1& registry) {
    std::vector<std::uint8_t> out(std::begin(magic), std::end(magic));
    append_u64(out, registry.candidates.size());
    for (const auto& candidate : registry.candidates) {
        append_u64(out, candidate.candidate_identity);
        append_u64(out, candidate.provider_identity);
        append_u64(out, candidate.operation_identity);
        out.push_back(static_cast<std::uint8_t>(candidate.origin));
        append_u32(out, candidate.provided_protocols);
        append_u32(out, candidate.required_protocols);
        out.push_back(static_cast<std::uint8_t>(candidate.missing_behavior));
        append_string(out, candidate.stable_name);
        append_string(out, candidate.source_locator);
        append_string(out, candidate.entry_symbol);
        append_bytes(out, candidate.opaque_payload.data(), candidate.opaque_payload.size());
    }
    return out;
}

custom_candidate_registration_code_v1 read_custom_candidate_binary_ir_v1(
    const std::vector<std::uint8_t>& bytes,
    custom_candidate_registry_v1* registry) {
    if (registry == nullptr || bytes.size() < sizeof(magic) ||
        !std::equal(std::begin(magic), std::end(magic), bytes.begin()))
        return custom_candidate_registration_code_v1::malformed_ir;
    reader in{bytes, sizeof(magic)};
    std::uint64_t count = 0u;
    if (!in.u64(&count) || count > bytes.size())
        return custom_candidate_registration_code_v1::malformed_ir;
    custom_candidate_registry_v1 decoded;
    for (std::uint64_t i = 0u; i < count; ++i) {
        custom_candidate_registration_v1 candidate{};
        std::uint32_t provided = 0u, required = 0u;
        if (!in.u64(&candidate.candidate_identity) || !in.u64(&candidate.provider_identity) ||
            !in.u64(&candidate.operation_identity) || in.offset >= bytes.size())
            return custom_candidate_registration_code_v1::malformed_ir;
        candidate.origin = static_cast<custom_candidate_origin_v1>(bytes[in.offset++]);
        if (!in.u32(&provided) || !in.u32(&required) || in.offset >= bytes.size())
            return custom_candidate_registration_code_v1::malformed_ir;
        candidate.provided_protocols = provided;
        candidate.required_protocols = required;
        candidate.missing_behavior =
            static_cast<missing_protocol_behavior_v1>(bytes[in.offset++]);
        if (!in.string(&candidate.stable_name) || !in.string(&candidate.source_locator) ||
            !in.string(&candidate.entry_symbol) || !in.blob(&candidate.opaque_payload))
            return custom_candidate_registration_code_v1::malformed_ir;
        const auto code = register_custom_candidate_v1(&decoded, std::move(candidate));
        if (code != custom_candidate_registration_code_v1::ok) return code;
    }
    if (in.offset != bytes.size()) return custom_candidate_registration_code_v1::malformed_ir;
    *registry = std::move(decoded);
    return custom_candidate_registration_code_v1::ok;
}

std::string write_custom_candidate_text_ir_v1(const custom_candidate_registry_v1& registry) {
    const auto binary = write_custom_candidate_binary_ir_v1(registry);
    std::string out = "ce.custom-candidates.v1 ";
    out.reserve(out.size() + binary.size() * 2u);
    for (const auto value : binary) {
        out.push_back(hex[value >> 4u]);
        out.push_back(hex[value & 15u]);
    }
    return out;
}

custom_candidate_registration_code_v1 read_custom_candidate_text_ir_v1(
    const std::string& text,
    custom_candidate_registry_v1* registry) {
    constexpr const char prefix[] = "ce.custom-candidates.v1 ";
    const std::string body = text.substr(0u, sizeof(prefix) - 1u) == prefix
        ? text.substr(sizeof(prefix) - 1u) : std::string{};
    if (body.empty() || body.size() % 2u != 0u)
        return custom_candidate_registration_code_v1::malformed_ir;
    std::vector<std::uint8_t> binary;
    binary.reserve(body.size() / 2u);
    for (std::size_t i = 0u; i < body.size(); i += 2u) {
        const int high = nibble(body[i]), low = nibble(body[i + 1u]);
        if (high < 0 || low < 0) return custom_candidate_registration_code_v1::malformed_ir;
        binary.push_back(static_cast<std::uint8_t>((high << 4) | low));
    }
    return read_custom_candidate_binary_ir_v1(binary, registry);
}

bool equivalent_custom_candidate_registries_v1(
    const custom_candidate_registry_v1& lhs,
    const custom_candidate_registry_v1& rhs) noexcept {
    return lhs.candidates.size() == rhs.candidates.size() &&
        std::equal(lhs.candidates.begin(), lhs.candidates.end(), rhs.candidates.begin(), same);
}

}  // namespace Cellerator::compiler::planning
