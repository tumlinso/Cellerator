#pragma once

#include <Cellerator/compiler/backend/nvcc/freeze_the_nvcc_backend_contract_v1.hh>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cellerator::compiler::backend::nvcc::v1 {

enum class cuda_entity_kind : std::uint8_t {
    constant = 1u,
    projection_view,
    device_helper,
    kernel,
    stage_launcher,
    host_stub,
    runtime_binding,
};

struct realized_cuda_entity {
    cuda_entity_kind kind = cuda_entity_kind::kernel;
    std::string stable_name;
    std::string declaration;
    std::uint32_t cellerator_line = 0u;
};

struct realized_cuda_module {
    std::string generated_path;
    std::string cellerator_path;
    std::vector<std::string> includes;
    std::vector<realized_cuda_entity> entities;
};

struct emitted_cuda_source {
    std::string text;
    std::vector<source_map_entry> source_map;
};

enum class emission_status : std::uint8_t {
    ok = 0u,
    invalid_module,
    invalid_entity,
    duplicate_entity,
};

[[nodiscard]] std::optional<emitted_cuda_source> emit_cuda_source(
    const realized_cuda_module& module, emission_status* status = nullptr) noexcept;

}  // namespace cellerator::compiler::backend::nvcc::v1
