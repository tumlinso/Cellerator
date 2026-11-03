#include <Cellerator/execution/dependency/transfer.cuh>

#include <exception>
#include <stdexcept>

namespace cellerator::execution::dependency {
namespace mc = ::cellerator::memory::cuda;
namespace {

bool select_peer_path(int source_device, int destination_device,
                      TransferMode mode) {
    if (source_device < 0 || destination_device < 0)
        throw std::invalid_argument("transfer devices must be nonnegative");
    if (source_device == destination_device)
        throw std::invalid_argument("prepared transfer requires distinct devices");

    switch (mode) {
    case TransferMode::staged:
        return false;
    case TransferMode::peer_if_supported:
    case TransferMode::peer_required:
        break;
    default:
        throw std::invalid_argument("unknown transfer mode");
    }

    int can_access_source = 0;
    {
        mc::DeviceGuard guard(destination_device);
        mc::check(cudaDeviceCanAccessPeer(&can_access_source,
                                          destination_device,
                                          source_device),
                  "cudaDeviceCanAccessPeer");
        if (!can_access_source) {
            if (mode == TransferMode::peer_required)
                throw std::runtime_error(
                    "peer transfer required but destination cannot access source");
            return false;
        }

        // The destination stream issues the peer copy, so only destination ->
        // source memory access is needed. This setting is device-wide and is
        // intentionally left enabled after this resource is destroyed.
        const cudaError_t status =
            cudaDeviceEnablePeerAccess(source_device, 0);
        if (status == cudaErrorPeerAccessAlreadyEnabled) {
            // The status is expected, but CUDA also leaves it in the thread's
            // last-error slot. Clear the handled result for later launch checks.
            (void)cudaGetLastError();
        } else if (status != cudaSuccess) {
            mc::check(status, "cudaDeviceEnablePeerAccess");
        }
    }
    return true;
}

void drain_stream_or_terminate(cudaStream_t stream, int device) noexcept {
    try {
        mc::DeviceGuard guard(device);
        mc::check(cudaStreamSynchronize(stream), "cudaStreamSynchronize after transfer failure");
    } catch (...) {
        // Continuing could free staging memory or let an owner mutate data
        // while a transfer is still using it. Fail closed if CUDA cannot drain.
        std::terminate();
    }
}

} // namespace

PreparedTransfer::PreparedTransfer(int source_device, int destination_device,
                                   std::size_t capacity, TransferMode mode)
    : source_device_(source_device),
      destination_device_(destination_device),
      capacity_(capacity),
      uses_peer_(select_peer_path(source_device, destination_device, mode)),
      staging_(uses_peer_ ? 0 : capacity),
      source_complete_(source_device),
      completion_(destination_device) {}

PreparedTransfer::~PreparedTransfer() noexcept {
    if (completion_valid_) {
        try {
            completion_.synchronize();
            return;
        } catch (...) {
            drain_last_streams();
            return;
        }
    }
    if (last_streams_valid_) drain_last_streams();
}

void PreparedTransfer::drain_last_streams() const noexcept {
    if (!last_streams_valid_) return;
    drain_stream_or_terminate(last_source_stream_, source_device_);
    drain_stream_or_terminate(last_destination_stream_, destination_device_);
}

void PreparedTransfer::enqueue(mc::ConstDeviceView source,
                               mc::DeviceView destination,
                               std::size_t bytes,
                               const mc::Stream& source_stream,
                               const mc::Stream& destination_stream) {
    if (bytes > capacity_)
        throw std::out_of_range("transfer exceeds prepared capacity");
    mc::require_view(source, bytes, source_device_, "transfer source");
    mc::require_view(destination, bytes, destination_device_, "transfer destination");
    if (source_stream.device() != source_device_)
        throw std::invalid_argument("source stream/device mismatch");
    if (destination_stream.device() != destination_device_)
        throw std::invalid_argument("destination stream/device mismatch");

    if (completion_valid_) {
        bool prior_ready = false;
        try {
            prior_ready = completion_.ready();
        } catch (...) {
            const auto failure = std::current_exception();
            if (last_streams_valid_) drain_last_streams();
            completion_valid_ = false;
            last_streams_valid_ = false;
            std::rethrow_exception(failure);
        }
        if (!prior_ready)
            throw std::logic_error("prepared transfer is already in flight");
    }

    // From here onward a failed event record can leave an operation in either
    // stream. Remember their borrowed handles so the error path and destructor
    // can drain them before staging or this object is released.
    last_source_stream_ = source_stream.get();
    last_destination_stream_ = destination_stream.get();
    last_streams_valid_ = true;
    completion_valid_ = false;

    bool source_touched = false;
    bool destination_touched = false;
    try {
        if (bytes == 0) {
            destination_touched = true;
            completion_.record(destination_stream);
            completion_valid_ = true;
            return;
        }

        if (!uses_peer_) {
            mc::DeviceGuard source_guard(source_device_);
            source_touched = true;
            mc::check(cudaMemcpyAsync(staging_.data(), source.data, bytes,
                                      cudaMemcpyDeviceToHost,
                                      source_stream.get()),
                      "cudaMemcpyAsync device-to-pinned");
        }

        // In the peer path this event waits for all source producers. In the
        // staged path it follows the D2H copy into the reusable pinned buffer.
        source_touched = true;
        source_complete_.record(source_stream);

        mc::DeviceGuard destination_guard(destination_device_);
        destination_touched = true;
        destination_stream.wait(source_complete_);
        if (uses_peer_) {
            mc::check(cudaMemcpyPeerAsync(destination.data, destination_device_,
                                          source.data, source_device_, bytes,
                                          destination_stream.get()),
                      "cudaMemcpyPeerAsync");
        } else {
            mc::check(cudaMemcpyAsync(destination.data, staging_.data(), bytes,
                                      cudaMemcpyHostToDevice,
                                      destination_stream.get()),
                      "cudaMemcpyAsync pinned-to-device");
        }
        completion_.record(destination_stream);
        completion_valid_ = true;
    } catch (...) {
        const auto failure = std::current_exception();
        if (source_touched)
            drain_stream_or_terminate(source_stream.get(), source_device_);
        if (destination_touched)
            drain_stream_or_terminate(destination_stream.get(), destination_device_);
        completion_valid_ = false;
        last_streams_valid_ = false;
        std::rethrow_exception(failure);
    }
}

const mc::Event& PreparedTransfer::completion() const {
    if (!completion_valid_)
        throw std::logic_error("prepared transfer has no successful enqueue");
    return completion_;
}

bool PreparedTransfer::ready() const {
    return completion_valid_ && completion_.ready();
}

void PreparedTransfer::wait() {
    if (!completion_valid_)
        throw std::logic_error("prepared transfer has no successful enqueue");
    try {
        completion_.synchronize();
    } catch (...) {
        const auto failure = std::current_exception();
        if (last_streams_valid_) drain_last_streams();
        completion_valid_ = false;
        last_streams_valid_ = false;
        std::rethrow_exception(failure);
    }
}

} // namespace cellerator::execution::dependency
