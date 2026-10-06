#pragma once

#include <Cellerator/compute/operation/indexed_mechanism/training.hh>

#include <cstdint>
#include <memory>
#include <mutex>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cellerator::bindings {

// Torch-free lifetime handle for one native indexed-mechanism owner/program.
// Tensor/framework adapters borrow these shared owners; they never rebuild the
// declaration or create a second parameter store.
class MechanismHandle final {
public:
    MechanismHandle(compute::operation::indexed::mechanism_declaration declaration,
                    std::span<const float> initial_coefficients, int device) {
        if (declaration.coefficients.extent != initial_coefficients.size())
            throw std::invalid_argument("initial coefficients do not match coefficient axis extent");
        owner_ = std::make_shared<compute::operation::indexed::mechanism_parameter_owner>(
            declaration.coefficients, declaration.coefficient_ids, initial_coefficients, device);
        program_ = std::make_shared<compute::operation::indexed::prepared_mechanism_program>(
            std::move(declaration), owner_);
    }
    ~MechanismHandle() = default;
    MechanismHandle(const MechanismHandle&) = delete;
    MechanismHandle& operator=(const MechanismHandle&) = delete;

    const compute::operation::indexed::mechanism_declaration& declaration() const { return program_->declaration(); }
    const std::shared_ptr<compute::operation::indexed::mechanism_parameter_owner>& owner() const { return owner_; }
    const std::shared_ptr<compute::operation::indexed::prepared_mechanism_program>& program() const { return program_; }

    std::vector<float> snapshot(void* stream = nullptr) const { return owner_->snapshot(stream); }
    void restore(std::span<const float> values, void* stream = nullptr) { owner_->restore(values, stream); }
    void preflight_write() const { owner_->preflight_write(); }
    void begin_write(void* stream = nullptr) { owner_->begin_write(stream); }
    void publish_write(void* stream = nullptr) { owner_->publish_write(stream); }
    void poison() noexcept { owner_->poison(); }
    bool owns_storage(std::uintptr_t address, std::size_t bytes) const noexcept {
        if (!bytes || address > std::numeric_limits<std::uintptr_t>::max() - bytes) return false;
        const auto begin = reinterpret_cast<std::uintptr_t>(owner_->data());
        const auto owner_bytes = owner_->size() * sizeof(float);
        if (begin > std::numeric_limits<std::uintptr_t>::max() - owner_bytes) return false;
        const auto end = begin + owner_bytes;
        return address < end && begin < address + bytes;
    }

    // Optional framework adapters may retain per-handle state without adding
    // framework types or dependencies to this native owner.
    std::shared_ptr<void> binding_adapter_state() const {
        std::lock_guard<std::mutex> lock(adapter_state_mutex_);
        return adapter_state_;
    }
    void set_binding_adapter_state(std::shared_ptr<void> state) {
        std::lock_guard<std::mutex> lock(adapter_state_mutex_);
        adapter_state_ = std::move(state);
    }

private:
    std::shared_ptr<compute::operation::indexed::mechanism_parameter_owner> owner_;
    std::shared_ptr<compute::operation::indexed::prepared_mechanism_program> program_;
    mutable std::mutex adapter_state_mutex_;
    std::shared_ptr<void> adapter_state_;
};

} // namespace cellerator::bindings
