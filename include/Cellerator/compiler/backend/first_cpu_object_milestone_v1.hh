#pragma once

#include <Cellerator/compiler/backend/compile_generated_c_into_ordinary_objects_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compiler::backend::v1 {

struct cpu_object_pipeline_receipt_v1 {
    std::uint64_t source_hash = 0;
    std::uint64_t syntax_ir_hash = 0;
    std::uint64_t semantic_ir_hash = 0;
    std::uint64_t planning_ir_hash = 0;
    std::uint64_t realization_ir_hash = 0;
};

struct first_cpu_object_request_v1 {
    std::string cell_source_path;
    std::string profile_name;
    std::string symbol;
    std::string compiler;
    std::string output_directory;
    cpu_object_pipeline_receipt_v1 pipeline{};
    std::vector<std::uint64_t> destination_offsets;
    std::vector<std::uint64_t> source_indices;
    std::vector<float> relation_values;
};

struct first_cpu_object_receipt_v1 {
    std::string generated_source_path;
    std::string object_path;
    compile_generated_cpp_receipt_v1 compilation{};
    cpu_object_pipeline_receipt_v1 pipeline{};
    bool profile_bound = false;
    bool source_to_native_provenance = false;
};

[[nodiscard]] compile_object_status_v1 deliver_first_cpu_object_v1(
    const first_cpu_object_request_v1& request,
    first_cpu_object_receipt_v1* receipt) noexcept;

}  // namespace cellerator::compiler::backend::v1
