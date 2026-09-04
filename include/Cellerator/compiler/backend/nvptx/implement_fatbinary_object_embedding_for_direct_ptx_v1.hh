#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::backend::nvptx {

enum class embedded_cuda_image_kind_v1 : std::uint8_t {
    ptx = 1u,
    cubin,
    fatbinary,
};

struct cuda_object_embedding_plan_v1 {
    embedded_cuda_image_kind_v1 image_kind = embedded_cuda_image_kind_v1::cubin;
    std::string executable;
    std::vector<std::string> arguments;
    std::string object_path;
    std::string section_name;
    std::string start_symbol;
    std::string end_symbol;
};

[[nodiscard]] cuda_object_embedding_plan_v1 make_cuda_object_embedding_plan_v1(
    const std::string& objcopy_executable,
    const std::string& image_basename,
    const std::string& object_path,
    embedded_cuda_image_kind_v1 image_kind);

struct embedded_cuda_image_v1 {
    const unsigned char* begin = nullptr;
    const unsigned char* end = nullptr;
    embedded_cuda_image_kind_v1 kind = embedded_cuda_image_kind_v1::cubin;
    std::string kernel_symbol;
};

struct embedded_cuda_launch_v1 {
    std::uint32_t grid_x = 1u;
    std::uint32_t grid_y = 1u;
    std::uint32_t grid_z = 1u;
    std::uint32_t block_x = 1u;
    std::uint32_t block_y = 1u;
    std::uint32_t block_z = 1u;
    std::uint32_t dynamic_shared_bytes = 0u;
    void* stream = nullptr;
    std::vector<void*> arguments;
};

enum class embedded_cuda_launch_status_v1 : std::uint8_t {
    success = 0u,
    invalid_image,
    invalid_launch,
    driver_unavailable,
    driver_error,
};

struct embedded_cuda_launch_result_v1 {
    embedded_cuda_launch_status_v1 status = embedded_cuda_launch_status_v1::invalid_image;
    int driver_code = 0;
    std::size_t image_bytes = 0u;
    std::string diagnostic;

    explicit operator bool() const noexcept {
        return status == embedded_cuda_launch_status_v1::success;
    }
};

// Loads an object-embedded PTX/cubin/fatbinary through the stable CUDA driver
// ABI and launches one named kernel. The image remains ordinary host object data.
[[nodiscard]] embedded_cuda_launch_result_v1 launch_embedded_cuda_image_v1(
    const embedded_cuda_image_v1& image,
    const embedded_cuda_launch_v1& launch);

}  // namespace Cellerator::compiler::backend::nvptx
