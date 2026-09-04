#include <Cellerator/compiler/sema/field/implement_field_level_reflection_identity_v1.hh>

#include <array>
#include <charconv>
#include <string_view>
#include <utility>

namespace Cellerator::compiler::sema::field {
namespace {

void append_hex(std::string& output, std::uint64_t value) {
    std::array<char, 16> digits{};
    const auto converted = std::to_chars(digits.data(), digits.data() + digits.size(), value, 16);
    const auto count = static_cast<std::size_t>(converted.ptr - digits.data());
    output.append(digits.size() - count, '0');
    output.append(digits.data(), count);
}

}  // namespace

field_reflection_identity_status_v1 implement_field_level_reflection_identity_v1(
    const execution_field_semantics_v1& field,
    field_reflection_identity_v1* identity) noexcept {
    if (identity == nullptr) return field_reflection_identity_status_v1::invalid_output;
    if ((field.identity.low == 0 && field.identity.high == 0) ||
        field.stable_source_name.empty() || !field.source.valid()) {
        return field_reflection_identity_status_v1::invalid_field;
    }

    field_reflection_identity_v1 result;
    result.field_identity = field.identity;
    result.stable_export_name = "cellerator.field.v1.";
    append_hex(result.stable_export_name, field.identity.high);
    result.stable_export_name.push_back('.');
    append_hex(result.stable_export_name, field.identity.low);
    *identity = std::move(result);
    return field_reflection_identity_status_v1::success;
}

}  // namespace Cellerator::compiler::sema::field
