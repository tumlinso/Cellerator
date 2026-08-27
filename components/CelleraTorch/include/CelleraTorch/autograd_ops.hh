#pragma once

#include <Cellerator/execution/training_program.hh>

#include <torch/torch.h>

#include <memory>
#include <mutex>

namespace celleratorch::autograd {

// One adapter binding describes one native Cellerator training launch. It owns
// no native storage. The caller-provided lifetime token must own every native
// pointer referenced by program and launch until backward finishes.
class native_training_binding final {
public:
    native_training_binding(
        cellerator::execution::training_program *program,
        cellerator::execution::training_program_launch launch,
        std::weak_ptr<void> native_lifetime) noexcept;

    native_training_binding(const native_training_binding &) = delete;
    native_training_binding &operator=(const native_training_binding &) = delete;

    bool has_result() const;
    cellerator::execution::training_program_result last_result() const;

private:
    friend class native_training_autograd_function;

    cellerator::execution::training_program *program_ = nullptr;
    cellerator::execution::training_program_launch launch_{};
    std::weak_ptr<void> native_lifetime_{};
    mutable std::mutex result_mutex_{};
    cellerator::execution::training_program_result result_{};
    bool has_result_ = false;
};

// The native training program is deliberately a combined N=16
// forward/backward/update step. This adapter therefore accepts the already
// computed Torch-visible forward output and invokes the native step exactly
// once from autograd backward. Cellerator updates relation values and bias;
// Torch receives only the dense-input gradient and must not optimize aliases of
// those native parameters a second time.
torch::Tensor native_training_autograd(
    const torch::Tensor &native_input,
    const torch::Tensor &forward_output,
    const torch::Tensor &training_output_storage,
    const torch::Tensor &input_gradient_storage,
    const std::shared_ptr<native_training_binding> &binding);

} // namespace celleratorch::autograd
