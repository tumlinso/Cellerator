#pragma once

#include <Cellerator/execution/conditional/parameter_store.hh>
#include <Cellerator/modeling/cuda/runtime.hh>
#include <memory>

namespace cellerator::modeling::cuda {

struct ParameterTransferMemory {
    std::size_t retained_source_bytes = 0;
    std::size_t pinned_staging_bytes = 0;
    std::size_t device_bytes = 0;
};

namespace detail { struct ParameterTransferState; struct ParameterUseState; }
class ParameterExecutionLease;

// Byte-exact H2D transfer of one already selected immutable parameter revision.
// Copies share ownership; destruction of the last owner drains its transfer.
// No dtype conversion, cache policy, replacement revision or optimizer transfer.
class ParameterTransfer final {
public:
    using ExpertKey = execution::conditional::ExpertKey;
    using ParameterLease = execution::conditional::ParameterLease;

    // Pure host validation/accounting: does not call CUDA. Source device must
    // be exactly "host"; pointers/immutability remain the ParameterStore contract.
    static ParameterTransferMemory required_memory(const ParameterLease& source);
    static ParameterTransfer upload(ParameterLease source, const Stream& stream);
    static ParameterTransfer upload(const execution::conditional::ParameterStore& store,
                                    ExpertKey exact, const Stream& stream);

    ExpertKey key() const;
    std::string dtype() const;
    ParameterTransferMemory memory() const;
    bool ready() const;
    void wait() const;
    // Requires transfer completion. The returned view borrows this ticket's
    // ownership; use acquire_on() to retain it through asynchronous consumers.
    ConstDeviceView view() const;
    ParameterExecutionLease acquire_on(const Stream& consumer) const;

private:
    friend class ParameterExecutionLease;
    explicit ParameterTransfer(std::shared_ptr<detail::ParameterTransferState> state);
    const detail::ParameterTransferState& checked() const;
    std::shared_ptr<detail::ParameterTransferState> state_;
};

// A move-only execution pin, separate from upload completion. acquire_on inserts
// a stream wait for upload. Submit all consumers on that SAME stream, then call
// finish() to record their completion. Keep the borrowed stream live until then.
// Destruction drains the event, or the stream if finish() was omitted, before
// freeing any parameter storage. Destructor drain errors terminate rather than
// free storage whose asynchronous lifetime cannot be established.
class ParameterExecutionLease final {
public:
    ~ParameterExecutionLease() noexcept;
    ParameterExecutionLease(ParameterExecutionLease&&) noexcept;
    ParameterExecutionLease& operator=(ParameterExecutionLease&&) noexcept;
    ParameterExecutionLease(const ParameterExecutionLease&) = delete;
    ParameterExecutionLease& operator=(const ParameterExecutionLease&) = delete;
    execution::conditional::ExpertKey key() const;
    ConstDeviceView view() const; // Refuses new view access after finish().
    void finish();               // Idempotent; seals this execution lease.
    bool ready() const;          // False until finish() and completion.
    void wait();                // Records finish if needed, then waits.
private:
    friend class ParameterTransfer;
    explicit ParameterExecutionLease(std::unique_ptr<detail::ParameterUseState> state);
    detail::ParameterUseState& checked() const;
    std::unique_ptr<detail::ParameterUseState> state_;
};

} // namespace cellerator::modeling::cuda
