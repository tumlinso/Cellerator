#pragma once

#include <Cellerator/compiler/lto/freeze_the_ceir_companion_object_artifact_contract_v1.hh>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compiler::lto::v1 {

enum class linker_input_kind_v1 : std::uint8_t {
    object = 1,
    static_archive_member,
    shared_library_metadata,
    sidecar
};

struct ceir_linker_input_v1 {
    linker_input_kind_v1 kind = linker_input_kind_v1::object;
    std::string path;
    std::string member;
    std::size_t bytes_scanned = 0;
    ceir_companion_artifact_v1 companion{};
};

struct extracted_ceir_record_v1 {
    artifact_identity_v1 identity{};
    artifact_identity_v1 profile{};
    std::string symbol;
    std::string source_path;
    std::string archive_member;
};

struct ceir_extraction_index_v1 {
    std::vector<extracted_ceir_record_v1> fields;
    std::size_t bytes_scanned = 0;
    std::size_t duplicate_members = 0;
    std::size_t peak_index_bytes = 0;
    bool loaded_native_code = false;
};

enum class ceir_extraction_status_v1 : std::uint8_t {
    valid = 0,
    invalid_companion,
    conflicting_duplicate
};

[[nodiscard]] ceir_extraction_status_v1 extract_ceir_linker_inputs_v1(
    const std::vector<ceir_linker_input_v1>& inputs,
    ceir_extraction_index_v1* index) noexcept;

}  // namespace cellerator::compiler::lto::v1
