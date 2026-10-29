#include <Cellerator/modeling/cuda/parameter_transfer.hh>
#include <cstring>
#include <exception>
#include <limits>

namespace cellerator::modeling::cuda {
namespace {
template<class F> void drain_or_terminate(F&& drain) noexcept {
    try { drain(); } catch (...) { std::terminate(); }
}
}
namespace detail {
struct ParameterTransferState {
    execution::conditional::ParameterLease source;
    PinnedBuffer staging;
    DeviceBuffer device;
    Event uploaded;
    ParameterTransferState(execution::conditional::ParameterLease lease, int target)
        : source(std::move(lease)), staging(source.buffers().parameters.bytes),
          device(source.buffers().parameters.bytes, target), uploaded(target) {}
    ~ParameterTransferState() noexcept {
        if (uploaded.recorded()) drain_or_terminate([&] { uploaded.synchronize(); });
    }
};

struct ParameterUseState {
    std::shared_ptr<ParameterTransferState> transfer;
    Stream stream;
    Event completed;
    bool finished = false;
    ParameterUseState(std::shared_ptr<ParameterTransferState> owner, const Stream& consumer)
        : transfer(std::move(owner)), stream(Stream::borrow(consumer.get(), consumer.device())),
          completed(consumer.device()) {
        stream.wait(transfer->uploaded);
    }
    ~ParameterUseState() noexcept {
        drain_or_terminate([&] {
            if (completed.recorded()) completed.synchronize();
            else stream.synchronize();
        });
    }
};
}

ParameterTransferMemory ParameterTransfer::required_memory(const ParameterLease& source) {
    const auto& buffer = source.buffers().parameters;
    buffer.validate();
    if (buffer.device != "host") throw std::invalid_argument("parameter upload requires an explicit host source");
    const auto& buffers = source.buffers();
    std::size_t retained = 0;
    for (auto bytes : {buffer.bytes, buffers.master_parameters.bytes,
                       buffers.gradients.bytes, buffers.optimizer_state.bytes}) {
        if (bytes > std::numeric_limits<std::size_t>::max() - retained)
            throw std::overflow_error("retained parameter byte accounting overflow");
        retained += bytes;
    }
    return {retained, buffer.bytes, buffer.bytes};
}
ParameterTransfer ParameterTransfer::upload(ParameterLease source, const Stream& stream) {
    const auto budget = required_memory(source); // Before allocation or CUDA initialization.
    auto state = std::make_shared<detail::ParameterTransferState>(std::move(source), stream.device());
    if (budget.pinned_staging_bytes)
        std::memcpy(state->staging.data(), state->source.buffers().parameters.data, budget.pinned_staging_bytes);
    try {
        state->device.upload_async(state->staging.data(), budget.device_bytes, stream);
        state->uploaded.record(stream);
    } catch (...) {
        // An enqueue can succeed even if recording its completion fails. Never
        // unwind pinned staging/device/source ownership before draining it.
        drain_or_terminate([&] { stream.synchronize(); });
        throw;
    }
    return ParameterTransfer(std::move(state));
}
ParameterTransfer ParameterTransfer::upload(const execution::conditional::ParameterStore& store,
                                             ExpertKey exact, const Stream& stream) {
    return upload(store.acquire(exact), stream);
}
ParameterTransfer::ParameterTransfer(std::shared_ptr<detail::ParameterTransferState> state)
    : state_(std::move(state)) {}
const detail::ParameterTransferState& ParameterTransfer::checked() const {
    if (!state_) throw std::logic_error("parameter transfer ticket was moved from");
    return *state_;
}
ParameterTransfer::ExpertKey ParameterTransfer::key() const { return checked().source.key(); }
std::string ParameterTransfer::dtype() const { return checked().source.buffers().parameters.dtype; }
ParameterTransferMemory ParameterTransfer::memory() const { return required_memory(checked().source); }
bool ParameterTransfer::ready() const { return checked().uploaded.ready(); }
void ParameterTransfer::wait() const { checked().uploaded.synchronize(); }
ConstDeviceView ParameterTransfer::view() const {
    if (!ready()) throw std::logic_error("parameter transfer has not completed");
    return checked().device.view();
}
ParameterExecutionLease ParameterTransfer::acquire_on(const Stream& consumer) const {
    if (checked().device.device() != consumer.device()) throw std::invalid_argument("parameter execution device mismatch");
    return ParameterExecutionLease(std::make_unique<detail::ParameterUseState>(state_, consumer));
}

ParameterExecutionLease::ParameterExecutionLease(std::unique_ptr<detail::ParameterUseState> state)
    : state_(std::move(state)) {}
ParameterExecutionLease::~ParameterExecutionLease() noexcept = default;
ParameterExecutionLease::ParameterExecutionLease(ParameterExecutionLease&&) noexcept = default;
ParameterExecutionLease& ParameterExecutionLease::operator=(ParameterExecutionLease&&) noexcept = default;
detail::ParameterUseState& ParameterExecutionLease::checked() const {
    if (!state_) throw std::logic_error("parameter execution lease was moved from");
    return *state_;
}
execution::conditional::ExpertKey ParameterExecutionLease::key() const { return checked().transfer->source.key(); }
ConstDeviceView ParameterExecutionLease::view() const {
    const auto& state = checked();
    if (state.finished) throw std::logic_error("execution lease is sealed");
    return state.transfer->device.view();
}
void ParameterExecutionLease::finish() {
    auto& state = checked();
    if (!state.finished) { state.completed.record(state.stream); state.finished = true; }
}
bool ParameterExecutionLease::ready() const {
    const auto& state = checked();
    return state.finished && state.completed.ready();
}
void ParameterExecutionLease::wait() { finish(); checked().completed.synchronize(); }

} // namespace cellerator::modeling::cuda
