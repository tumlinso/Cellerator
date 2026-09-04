#include <Cellerator/compiler/ir/semantic/implement_semantic_canonicalization_and_equivalence_v1.hh>

#include <algorithm>
#include <cctype>
#include <string_view>
#include <type_traits>

namespace Cellerator::compiler::ir::semantic {
namespace {

bool is_semantic_punctuation(char value) noexcept {
    switch (value) {
        case ':': case ',': case '.': case '(': case ')': case '<': case '>':
        case '[': case ']': case '{': case '}': case '*': case '&': case '=':
            return true;
        default:
            return false;
    }
}

std::string normalize_operation_spelling(std::string_view spelling) {
    std::string result;
    result.reserve(spelling.size());
    bool pending_space = false;
    for (const unsigned char value : spelling) {
        if (std::isspace(value)) {
            pending_space = !result.empty();
            continue;
        }
        const auto character = static_cast<char>(value);
        if (is_semantic_punctuation(character)) {
            if (!result.empty() && result.back() == ' ') result.pop_back();
            result.push_back(character);
            pending_space = false;
            continue;
        }
        if (pending_space && !result.empty() && !is_semantic_punctuation(result.back()))
            result.push_back(' ');
        result.push_back(character);
        pending_space = false;
    }
    return result;
}

void set_status(semantic_canonicalization_status_v1* status,
                semantic_canonicalization_status_v1 value) noexcept {
    if (status != nullptr) *status = value;
}

bool valid_identity_list(const std::vector<semantic_identity_v1>& identities) noexcept {
    return std::all_of(identities.begin(), identities.end(),
                       [](const auto& identity) { return identity.valid(); });
}

bool equal_identity(semantic_identity_v1 left, semantic_identity_v1 right) noexcept {
    return left.low == right.low && left.high == right.high;
}

bool equal_identities(const std::vector<semantic_identity_v1>& left,
                      const std::vector<semantic_identity_v1>& right) noexcept {
    return left.size() == right.size() &&
        std::equal(left.begin(), left.end(), right.begin(), equal_identity);
}

bool equal_numeric(const numeric_tuple_ir_v1& left,
                   const numeric_tuple_ir_v1& right) noexcept {
    return left.storage == right.storage && left.compute == right.compute &&
        left.accumulation == right.accumulation && left.output == right.output;
}

class fingerprint_builder_v1 {
public:
    void byte(std::uint8_t value) noexcept {
        low_ = (low_ ^ value) * 1099511628211ULL;
        high_ ^= value;
        high_ *= 0x9e3779b185ebca87ULL;
        high_ ^= high_ >> 29;
    }

    template <typename Value>
    void scalar(Value value) noexcept {
        using unsigned_type = std::make_unsigned_t<Value>;
        auto encoded = static_cast<unsigned_type>(value);
        for (std::size_t index = 0; index < sizeof(encoded); ++index) {
            byte(static_cast<std::uint8_t>(encoded & 0xffu));
            encoded >>= 8u;
        }
    }

    void identity(semantic_identity_v1 value) noexcept {
        scalar(value.low);
        scalar(value.high);
    }

    void text(std::string_view value) noexcept {
        scalar(static_cast<std::uint64_t>(value.size()));
        for (const unsigned char character : value) byte(character);
    }

    [[nodiscard]] semantic_fingerprint_v1 finish() const noexcept {
        return {low_, high_};
    }

private:
    std::uint64_t low_ = 14695981039346656037ULL;
    std::uint64_t high_ = 0xd6e8feb86659fd93ULL;
};

}  // namespace

std::optional<semantic_canonical_record_v1> canonicalize_semantic_record_v1(
    semantic_canonical_record_v1 record,
    semantic_canonicalization_status_v1* status) noexcept {
    if (!record.operation_identity.valid()) {
        set_status(status, semantic_canonicalization_status_v1::invalid_operation_identity);
        return std::nullopt;
    }
    record.operation_spelling = normalize_operation_spelling(record.operation_spelling);
    if (record.operation_spelling.empty()) {
        set_status(status, semantic_canonicalization_status_v1::invalid_operation_spelling);
        return std::nullopt;
    }
    if (!valid_identity_list(record.input_types) ||
        !valid_identity_list(record.output_types)) {
        set_status(status, semantic_canonicalization_status_v1::invalid_type_identity);
        return std::nullopt;
    }
    if (!valid_identity_list(record.biological_identities)) {
        set_status(status, semantic_canonicalization_status_v1::invalid_biological_identity);
        return std::nullopt;
    }
    if (validate_numeric_tuple_ir_v1(record.numerical) !=
        state_value_ir_validation_code_v1::success) {
        set_status(status, semantic_canonicalization_status_v1::invalid_numerical_contract);
        return std::nullopt;
    }
    if (record.field_identity == 0) {
        set_status(status, semantic_canonicalization_status_v1::invalid_field_identity);
        return std::nullopt;
    }
    if (record.field_boundary != execution_field_boundary_ir_v1::transparent &&
        record.field_boundary != execution_field_boundary_ir_v1::explicit_boundary) {
        set_status(status, semantic_canonicalization_status_v1::invalid_field_boundary);
        return std::nullopt;
    }
    set_status(status, semantic_canonicalization_status_v1::success);
    return record;
}

std::optional<semantic_fingerprint_v1> fingerprint_semantic_record_v1(
    const semantic_canonical_record_v1& record,
    semantic_canonicalization_status_v1* status) noexcept {
    const auto canonical = canonicalize_semantic_record_v1(record, status);
    if (!canonical) return std::nullopt;

    fingerprint_builder_v1 builder;
    builder.text("Cellerator.semantic.v1");
    builder.identity(canonical->operation_identity);
    builder.text(canonical->operation_spelling);
    builder.scalar(static_cast<std::uint64_t>(canonical->input_types.size()));
    for (const auto identity : canonical->input_types) builder.identity(identity);
    builder.scalar(static_cast<std::uint64_t>(canonical->output_types.size()));
    for (const auto identity : canonical->output_types) builder.identity(identity);
    builder.scalar(static_cast<std::uint64_t>(canonical->biological_identities.size()));
    for (const auto identity : canonical->biological_identities) builder.identity(identity);
    builder.scalar(static_cast<std::uint8_t>(canonical->numerical.storage));
    builder.scalar(static_cast<std::uint8_t>(canonical->numerical.compute));
    builder.scalar(static_cast<std::uint8_t>(canonical->numerical.accumulation));
    builder.scalar(static_cast<std::uint8_t>(canonical->numerical.output));
    builder.scalar(canonical->effects);
    builder.scalar(canonical->field_identity);
    builder.scalar(static_cast<std::uint8_t>(canonical->field_boundary));
    return builder.finish();
}

bool semantic_equivalent_v1(const semantic_canonical_record_v1& left,
                            const semantic_canonical_record_v1& right) noexcept {
    const auto canonical_left = canonicalize_semantic_record_v1(left);
    const auto canonical_right = canonicalize_semantic_record_v1(right);
    if (!canonical_left || !canonical_right) return false;
    return equal_identity(canonical_left->operation_identity,
                          canonical_right->operation_identity) &&
        canonical_left->operation_spelling == canonical_right->operation_spelling &&
        equal_identities(canonical_left->input_types, canonical_right->input_types) &&
        equal_identities(canonical_left->output_types, canonical_right->output_types) &&
        equal_identities(canonical_left->biological_identities,
                         canonical_right->biological_identities) &&
        equal_numeric(canonical_left->numerical, canonical_right->numerical) &&
        canonical_left->effects == canonical_right->effects &&
        canonical_left->field_identity == canonical_right->field_identity &&
        canonical_left->field_boundary == canonical_right->field_boundary;
}

}  // namespace Cellerator::compiler::ir::semantic
