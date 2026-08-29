#pragma once

#include "domain.hh"
#include "status.hh"

#include <cuda_runtime.h>

#include <cstddef>

namespace cellerator::memory {

enum class copy_direction : unsigned char {
    host_to_device = 0,
    device_to_host,
    device_to_device,
    host_to_host
};

struct copy_request {
    void *destination = nullptr;
    std::size_t destination_capacity = 0;
    placement destination_where{};
    const void *source = nullptr;
    std::size_t source_capacity = 0;
    placement source_where{};
    std::size_t bytes = 0;
    copy_direction direction = copy_direction::host_to_host;
    cudaStream_t stream = nullptr;
};

status copy_async(const copy_request &request) noexcept;

} // namespace cellerator::memory
