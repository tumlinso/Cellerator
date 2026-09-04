#pragma once
#include <Cellerator/compiler/backend/nvcc/freeze_the_nvcc_backend_contract_v1.hh>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
namespace cellerator::compiler::backend::nvcc::v1 {
enum class compilation_route:std::uint8_t{whole_cuda=1,split_host_device};
struct split_request{std::uint64_t semantic_identity=0;std::string generated_cuda,generated_host,output;std::vector<std::uint32_t> architectures;compilation_route route=compilation_route::whole_cuda;};
struct compilation_graph{std::uint64_t semantic_identity=0;std::vector<compilation_job> jobs;};
enum class split_status:std::uint8_t{ok=0,invalid_identity,invalid_path,invalid_route,invalid_job};
[[nodiscard]] std::optional<compilation_graph> make_compilation_graph(const split_request&,split_status* = nullptr) noexcept;
}
