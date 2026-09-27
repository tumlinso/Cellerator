#pragma once

#include <Cellerator/compute/operation/indexed_mechanism/training.hh>

#include <torch/torch.h>
#include <torch/custom_class.h>

#include <c10/util/intrusive_ptr.h>
#include <memory>
#include <vector>

namespace celleratorch {

// Python and C++ share this serialized declaration. axis_words contains three
// persistent biological identities (input, output, coefficient), each encoded
// as domain/order/geometry/partition low/high words. axis_extents has 3 entries.
// The native IEEE conversion surface remains unrestricted; this adapter rejects
// mixed-mode masters that could overflow its derived FP16 execution plane.
class Mechanism final : public torch::CustomClassHolder {
public:
    Mechanism(torch::Tensor initial_coefficients,
        std::vector<std::int64_t> axis_words,
        std::vector<std::int64_t> axis_extents,
        std::vector<std::int64_t> coefficient_id_words,
        std::vector<std::int64_t> mechanism_id_words,
        std::vector<std::int64_t> arg_offsets,
        std::vector<std::int64_t> arg_slots,
        std::vector<std::int64_t> arg_role_words,
        std::vector<std::int64_t> arg_axes,
        std::vector<std::int64_t> arg_indices,
        std::vector<std::int64_t> coefficient_bindings,
        std::vector<std::int64_t> output_offsets,
        std::vector<std::int64_t> output_slots,
        std::vector<std::int64_t> output_role_words,
        std::vector<std::int64_t> output_axes,
        std::vector<std::int64_t> output_indices,
        std::vector<std::int64_t> output_assembly_words,
        std::vector<double> output_scales,
        std::int64_t max_batch, std::int64_t max_live_forwards,
        std::int64_t precision);

    torch::Tensor coefficients() const;
    void validate_coefficients(const torch::Tensor& coefficients) const;
    torch::Tensor forward(const torch::Tensor& input,
                          const std::vector<std::int64_t>& axis_words);
    void preflight_update() const;
    void begin_update();
    void publish_update();
    void poison();
    torch::Tensor snapshot();
    void restore(const torch::Tensor& coefficients);
    std::int64_t generation() const;
    bool poisoned() const;
    std::int64_t device() const;

    std::shared_ptr<cellerator::compute::operation::indexed::prepared_mechanism_program> program() const;
    std::shared_ptr<cellerator::compute::operation::indexed::mechanism_parameter_owner> owner() const;

private:
    struct implementation;
    std::shared_ptr<implementation> impl_;
};

// C++ module adapter. Constructing multiple modules from the same Mechanism
// registers the same leaf Tensor object and therefore accumulates one gradient.
class MechanismModule final : public torch::nn::Module {
public:
    explicit MechanismModule(c10::intrusive_ptr<Mechanism> mechanism);
    torch::Tensor forward(const torch::Tensor& input,
                          const std::vector<std::int64_t>& axis_words);
    c10::intrusive_ptr<Mechanism> mechanism() const;
private:
    c10::intrusive_ptr<Mechanism> mechanism_;
    torch::Tensor coefficients_;
};

// Runs stock Adam once after validating all CE owners and optimizer parameters.
// Tensor options with coupled weight decay are rejected for CE master values.
void guarded_adam_step(torch::optim::Optimizer& optimizer,
    const std::vector<c10::intrusive_ptr<Mechanism>>& mechanisms,
    bool scaler_skipped = false);

} // namespace celleratorch
