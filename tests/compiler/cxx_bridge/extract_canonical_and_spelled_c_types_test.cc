#include <Cellerator/compiler/frontend/cxx/extract_canonical_and_spelled_c_types_v1.hh>
#include <Cellerator/compiler/frontend/cxx/parse_shadow_translation_units_with_full_c_semantics_v1.hh>

#include <cstdint>
#include <iostream>
#include <string>

namespace cxx = Cellerator::compiler::frontend::cxx;

namespace downstream {
struct half_type { std::uint16_t storage; };
struct bf16_type { std::uint16_t storage; };
struct numeric { float x; int y; };
}

int main() {
    cxx::cxx_compilation_invocation_request_v1 invocation_request;
    invocation_request.clang_driver_path = "/usr/bin/clang++-18";
    invocation_request.target_triple = "x86_64-pc-linux-gnu";
    cxx::cxx_compilation_invocation_v1 invocation;
    if (cxx::create_cxx_compilation_invocation_v1(invocation_request, &invocation) !=
        cxx::cxx_compilation_invocation_status_v1::success) {
        return 1;
    }

    const std::string source = R"cpp(
struct __half { unsigned short storage; };
struct __nv_bfloat16 { unsigned short storage; };
using half_alias = __half;
using bf16_alias = __nv_bfloat16;
using float4 = float __attribute__((ext_vector_type(4)));
struct numeric { float x; int y; };
half_alias half_value{};
bf16_alias bf16_value{};
float4 vector_value{};
int* pointer_value{};
int base_value = 0;
int& reference_value = base_value;
int __attribute__((address_space(1))) * address_value;
numeric user_value{};
)cpp";
    cxx::shadow_translation_unit_request_v1 parse_request;
    parse_request.invocation = &invocation;
    parse_request.virtual_filename = "type_queries.cc";
    parse_request.source = source;
    cxx::shadow_translation_unit_v1 unit;
    if (cxx::parse_shadow_translation_unit_v1(parse_request, &unit) !=
        cxx::shadow_translation_unit_status_v1::success) {
        return 1;
    }

    const auto offset = [&source](const char* name) {
        return static_cast<std::uint32_t>(source.find(name));
    };
    std::vector<cxx::source_capture_request_v1> requests{
        {cxx::source_capture_kind_v1::state, "half_value", offset("half_value")},
        {cxx::source_capture_kind_v1::state, "bf16_value", offset("bf16_value")},
        {cxx::source_capture_kind_v1::state, "vector_value", offset("vector_value")},
        {cxx::source_capture_kind_v1::state, "pointer_value", offset("pointer_value")},
        {cxx::source_capture_kind_v1::state, "reference_value", offset("reference_value")},
        {cxx::source_capture_kind_v1::state, "address_value", offset("address_value")},
        {cxx::source_capture_kind_v1::state, "user_value", offset("user_value")},
    };
    cxx::source_capture_binding_result_v1 bindings;
    if (cxx::bind_source_captures_v1(
            cxx::source_capture_binding_schema_version_v1,
            unit.adapter(), requests, &bindings) != cxx::source_capture_binding_status_v1::success) {
        return 1;
    }
    std::vector<cxx::cxx_type_record_v1> records;
    if (cxx::extract_cxx_types_v1(
            cxx::cxx_type_extraction_schema_version_v1,
            unit.adapter(), bindings.captures, &records) !=
            cxx::cxx_type_extraction_status_v1::success || records.size() != requests.size()) {
        return 1;
    }

    const std::uint32_t expected_traits[] = {
        cxx::cxx_type_half_v1,
        cxx::cxx_type_bfloat16_v1,
        cxx::cxx_type_vector_v1,
        cxx::cxx_type_pointer_v1,
        cxx::cxx_type_lvalue_reference_v1,
        cxx::cxx_type_pointer_v1 | cxx::cxx_type_address_space_v1,
        cxx::cxx_type_user_defined_v1,
    };
    for (std::size_t index = 0; index < records.size(); ++index) {
        if ((records[index].traits & expected_traits[index]) != expected_traits[index] ||
            records[index].user_spelling.empty() || records[index].canonical_spelling.empty() ||
            records[index].canonical_identity.rfind("cxx-type-v1:", 0) != 0) {
            std::cerr << "type trait or identity mismatch at " << index
                      << ": traits=" << records[index].traits
                      << " user='" << records[index].user_spelling
                      << "' canonical='" << records[index].canonical_spelling << "'\n";
            return 1;
        }
    }
    if (records[0].user_spelling.find("half_alias") == std::string::npos ||
        records[0].canonical_spelling.find("__half") == std::string::npos ||
        records[0].size_bytes != sizeof(downstream::half_type) ||
        records[0].alignment_bytes != alignof(downstream::half_type) ||
        records[1].size_bytes != sizeof(downstream::bf16_type) ||
        records[2].size_bytes != 4 * sizeof(float) ||
        records[3].size_bytes != sizeof(void*) ||
        records[6].size_bytes != sizeof(downstream::numeric) ||
        records[6].alignment_bytes != alignof(downstream::numeric)) {
        std::cerr << "spelling or downstream ABI size/alignment mismatch\n";
        return 1;
    }
    return 0;
}
