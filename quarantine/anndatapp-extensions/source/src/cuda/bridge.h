#pragma once
#include <anndatapp/device_abi.h>
#include <cstdint>

// Return a cudaError_t value as an int. No CUDA declarations cross the module
// interface. The caller has already checked all metadata and set the device.
extern "C" int anndatapp_cuda_fill_bits(anndatapp::device::DenseView, std::uint64_t, void* stream);
extern "C" int anndatapp_cuda_fill_float(anndatapp::device::DenseView, double, void* stream);
extern "C" int anndatapp_cuda_cast(anndatapp::device::DenseView, anndatapp::device::DenseView,
                                 void* status, void* stream);
extern "C" int anndatapp_cuda_validate_sparse(anndatapp::device::SparseView, void* status, void* stream);
extern "C" int anndatapp_cuda_gather_columns(anndatapp::device::DenseView, anndatapp::device::DenseView,
    anndatapp::device::Span, anndatapp::device::ScalarType, void* status, void* stream);
