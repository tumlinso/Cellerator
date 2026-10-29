#pragma once

#include <Cellerator/execution/conditional/dispatch.hh>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>

namespace cellerator::execution::conditional {

// The owner retains allocation lifetime (including a backend-specific deleter).
// Published bytes are immutable: backends must allocate replacement buffers for
// an update, not mutate storage referenced by an existing revision or lease.
struct ParameterBuffer {
    std::shared_ptr<const void> owner;
    const void* data = nullptr;
    std::size_t bytes = 0;
    std::string dtype;
    std::string device = "host";
    void validate() const;
};

struct ParameterBuffers {
    ParameterBuffer parameters;
    ParameterBuffer master_parameters;
    ParameterBuffer gradients;
    ParameterBuffer optimizer_state;
    void validate() const;
};

struct ParameterRevision {
    ExpertKey key;
    ParameterBuffers buffers;
};

// Copyable RAII execution pin. It remains valid after the store is destroyed or
// a retired version is unregistered. Keep it through asynchronous device work
// and through backward when backward reads the forward parameters.
class ParameterLease final {
public:
    const ParameterRevision& revision() const;
    const ParameterBuffers& buffers() const { return revision().buffers; }
    ExpertKey key() const { return revision().key; }
    explicit operator bool() const noexcept { return bool(value_); }
    void release() noexcept { value_.reset(); }
private:
    friend class ParameterStore;
    explicit ParameterLease(std::shared_ptr<const ParameterRevision> value)
        : value_(std::move(value)) {}
    std::shared_ptr<const ParameterRevision> value_;
};

class StaleParameterRevision : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// One logical trainable store; residency and placement are separate services.
// Numeric dtype, reduction, precision and optimizer rules belong to the supplied
// update backend. No implicit conversion or gradient reduction occurs here.
class ParameterStore final {
public:
    using Update = std::function<ParameterBuffers(
        const ParameterRevision&, const ParameterBuffer& gradients)>;
    void register_initial(ExpertKey key, ParameterBuffers buffers);
    ExpertKey current(std::uint64_t expert_id) const;
    ParameterLease acquire(ExpertKey key) const;
    std::vector<ParameterLease> snapshot(const std::vector<ExpertKey>& keys) const;
    // Compare-and-publish exact base. Callback runs outside the store lock;
    // concurrent losers throw rather than overwrite or replay an update.
    ExpertKey update(ExpertKey expected, std::uint64_t new_revision,
                     ParameterBuffer gradients, const Update& backend);
    void publish(ExpertKey expected, ExpertKey next, ParameterBuffers buffers);
    // Current revisions cannot retire. Existing leases survive retirement;
    // retired identities may never be reused, preventing revision ABA.
    bool retire(ExpertKey key);
    std::size_t leases(ExpertKey key) const;
private:
    mutable std::mutex mutex_;
    std::map<ExpertKey, std::shared_ptr<const ParameterRevision>> revisions_;
    std::map<std::uint64_t, ExpertKey> current_;
    std::set<ExpertKey> used_;
};

} // namespace cellerator::execution::conditional
