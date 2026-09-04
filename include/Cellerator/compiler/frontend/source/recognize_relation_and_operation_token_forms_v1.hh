#pragma once

#include <Cellerator/compiler/frontend/source/define_the_unified_source_location_model_v1.hh>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::frontend::source {

enum class raw_operation_form_kind_v1 : std::uint8_t { relation_transfer = 1, intrinsic_call };
struct raw_operation_form_v1 {
    raw_operation_form_kind_v1 kind = raw_operation_form_kind_v1::relation_transfer;
    source_span_v1 span{};
    std::string payload;
};
struct raw_operation_scan_v1 { std::vector<raw_operation_form_v1> forms; bool recovered = true; };

[[nodiscard]] raw_operation_scan_v1 recognize_operation_forms_v1(
    source_space_id_v1 source, std::string_view bytes);

} // namespace Cellerator::compiler::frontend::source
