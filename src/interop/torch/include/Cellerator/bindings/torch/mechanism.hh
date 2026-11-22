#pragma once

#include <Cellerator/bindings/mechanism_handle.hh>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace cellerator::bindings::torch {

::torch::Tensor coefficients(const std::shared_ptr<MechanismHandle>& handle);
void validate_coefficients(const ::torch::Tensor& coefficients,
                           const std::shared_ptr<MechanismHandle>& handle);
::torch::Tensor mechanism_apply(const ::torch::Tensor& input,
                              const ::torch::Tensor& coefficients,
                              const std::shared_ptr<MechanismHandle>& handle,
                              const std::vector<std::int64_t>& axis_words);
bool is_native_coefficient(const ::torch::Tensor& tensor,
                           const std::vector<std::shared_ptr<MechanismHandle>>& handles);
void begin_update(const std::shared_ptr<MechanismHandle>& handle);
void publish_update(const std::shared_ptr<MechanismHandle>& handle);
void synchronize_coefficients(const std::shared_ptr<MechanismHandle>& handle);

} // namespace cellerator::bindings::torch
