#pragma once

#include <Cellerator/compute/operation/indexed_mechanism/incidence.hh>
#include <memory>
#include <span>
#include <vector>

namespace cellerator::compute::operation::indexed {

enum class training_precision { f32, mixed_f16 };
struct product_contribution {
    output_index destination{};
    float scale = 1;
};
struct product_mechanism {
    identity instance{};
    std::vector<argument_index> arguments;
    std::vector<product_contribution> outputs;
    std::uint64_t coefficient = 0;
};
struct mechanism_declaration {
    indexed_axis input{}, output{}, coefficients{};
    std::vector<identity> coefficient_ids;
    std::vector<product_mechanism> mechanisms;
    std::uint64_t max_batch = 0;
    std::uint32_t max_live_forwards = 8;
    training_precision precision = training_precision::f32;
};

// Session-owned canonical logical-order coefficients. Calls are host-serialized;
// events order asynchronous readers and the exclusive writer on caller streams.
class mechanism_parameter_owner {
public:
    mechanism_parameter_owner(indexed_axis axis, std::vector<identity> ids,
                              std::span<const float> initial, int device,
                              std::uint32_t reader_capacity = 32);
    ~mechanism_parameter_owner();
    mechanism_parameter_owner(const mechanism_parameter_owner&) = delete;
    mechanism_parameter_owner& operator=(const mechanism_parameter_owner&) = delete;
    float* data() const;
    const std::uint16_t* half_data() const;
    std::uint64_t generation() const;
    std::uint64_t size() const;
    int device() const;
    const indexed_axis& axis() const;
    const std::vector<identity>& logical_ids() const;
    bool poisoned() const;
    void preflight_write() const;
    void begin_write(void* stream);
    void publish_write(void* stream);
    void poison() noexcept;
    // Whole-checkpoint restoration only, including downstream optimizer state.
    void restore(std::span<const float> values, void* stream);
    std::vector<float> snapshot(void* stream) const;
    std::uint32_t acquire_reader(void* stream);
    void complete_reader(std::uint32_t ticket, void* stream) noexcept;
private:
    void poison_locked() noexcept;
    struct implementation;
    std::unique_ptr<implementation> impl_;
};

class prepared_mechanism_program;
class mechanism_tape {
public:
    ~mechanism_tape();
    mechanism_tape(const mechanism_tape&) = delete;
    mechanism_tape& operator=(const mechanism_tape&) = delete;
    std::uint64_t batch() const;
private:
    friend class prepared_mechanism_program;
    struct implementation;
    explicit mechanism_tape(std::unique_ptr<implementation>);
    std::unique_ptr<implementation> impl_;
};

// One input biological axis and one output axis, batched over independent cells.
// Every pointer passed to forward/backward must refer to device memory on the
// coefficient owner's device; output and VJP buffers are FP32. The caller owns
// dense buffers and tensor lifetimes. Input/output streams must use that device.
// Capacity is reserved during prepare.
class prepared_mechanism_program : public std::enable_shared_from_this<prepared_mechanism_program> {
public:
    prepared_mechanism_program(mechanism_declaration declaration,
                              std::shared_ptr<mechanism_parameter_owner> parameters);
    ~prepared_mechanism_program();
    const mechanism_declaration& declaration() const;
    const std::shared_ptr<mechanism_parameter_owner>& parameters() const;
    std::size_t reserved_bytes() const;
    std::shared_ptr<mechanism_tape> forward(
        const void* input, bool input_half, std::uint64_t batch,
        const execution::persistent_axis_identity& input_axis,
        float* output, void* stream);
    // First-order VJP; consumes the tape exactly once. Coefficient gradients are
    // FP32 logical order, input gradients accumulate in FP32.
    void backward(mechanism_tape& tape, const float* output_gradient,
                  float* input_gradient, float* coefficient_gradient, void* stream);
private:
    friend class mechanism_tape;
    struct implementation;
    std::unique_ptr<implementation> impl_;
};

} // namespace cellerator::compute::operation::indexed
