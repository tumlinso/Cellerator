#include "minimal_direct_layout.hh"

#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

namespace cellerator::bench::factorized_interactions::controls {
namespace {

struct Region {
    std::size_t offset = 0;
    std::size_t bytes = 0;
};

std::size_t checked_add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::overflow_error("minimal direct upload size overflows size_t");
    return a + b;
}

std::size_t checked_array_bytes(std::uint64_t count, std::size_t width) {
    if (width && count > std::numeric_limits<std::size_t>::max() / width)
        throw std::overflow_error("minimal direct upload array exceeds host size");
    return static_cast<std::size_t>(count) * width;
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    const auto remainder = value % alignment;
    return remainder ? checked_add(value, alignment - remainder) : value;
}

template<class T>
Region append_region(std::size_t& total, const std::vector<T>& values) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "canonical device records must be trivially copyable");
    if (values.empty()) return {};
    total = align_up(total, alignof(T));
    Region region{total, checked_array_bytes(
        static_cast<std::uint64_t>(values.size()), sizeof(T))};
    total = checked_add(total, region.bytes);
    return region;
}

template<class T>
void stage(void* destination, const std::vector<T>& values, Region region) {
    if (region.bytes)
        std::memcpy(static_cast<char*>(destination) + region.offset,
                    values.data(), region.bytes);
}

template<class T>
const T* device_pointer(const void* base, Region region) noexcept {
    return region.bytes
        ? reinterpret_cast<const T*>(static_cast<const char*>(base) + region.offset)
        : nullptr;
}

} // namespace

struct MinimalDirectDeviceLayout::State {
    struct Selection {
        modeling::cuda::FactorizedInteractionProblem problem{};
        int device = -1;
    };

    static Selection validate_selection(
        const MinimalDirectHostLayout& host_layout,
        modeling::cuda::FactorizedInteractionProblem problem,
        int device, const memory::cuda::Stream& upload_stream) {
        if (device < 0)
            throw std::invalid_argument("direct control device must be nonnegative");
        if (upload_stream.device() != device)
            throw std::invalid_argument("direct control upload stream/device mismatch");
        modeling::cuda::FactorizedInteractionBackend selected{};
        const auto status = host_layout.backend_for(problem, &selected);
        if (status != cudaSuccess)
            throw std::invalid_argument("direct control backend query rejected the problem");
        if (selected != modeling::cuda::FactorizedInteractionBackend::direct)
            throw std::invalid_argument(
                "minimal direct layout cannot serve the shared stable backend");
        return {problem, device};
    }

    State(const MinimalDirectHostLayout& host_layout,
          modeling::cuda::FactorizedInteractionProblem selected_problem,
          int selected_device,
          const memory::cuda::Stream& upload_stream)
        : State(host_layout,
                validate_selection(host_layout, selected_problem, selected_device,
                                   upload_stream),
                upload_stream) {}

    State(const MinimalDirectHostLayout& host_layout, Selection selection,
          const memory::cuda::Stream& upload_stream)
        : problem(selection.problem), device(selection.device),
          upload_stream_handle(upload_stream.get()), completion(selection.device, false) {
        auto host_view = host_layout.view();
        if (host_view.mechanism_count && !host_view.mechanisms)
            throw std::invalid_argument("direct host records are unavailable");
        if (host_view.term_count && !host_view.terms)
            throw std::invalid_argument("direct host terms are unavailable");

        std::size_t total = 0;
        const Region records_region = append_region(total, host_layout.mechanisms());
        const Region terms_region = append_region(total, host_layout.terms());
        storage_bytes = total;
        storage = memory::cuda::DeviceBuffer(storage_bytes, device);
        staging = memory::cuda::PinnedBuffer(storage_bytes);
        if (storage_bytes) {
            stage(staging.data(), host_layout.mechanisms(), records_region);
            stage(staging.data(), host_layout.terms(), terms_region);
            // One immutable bulk transfer keeps the cold control's upload
            // count directly comparable with the prepared owner.
            storage.upload_async(staging.data(), storage_bytes, upload_stream);
        }
        device_view = host_view;
        device_view.mechanisms = device_pointer<layout::FactorizedInteractionRecord>(
            storage.data(), records_region);
        device_view.terms = device_pointer<layout::FactorizedInteractionTermRecord>(
            storage.data(), terms_region);
        completion.record(upload_stream);
    }

    ~State() noexcept {
        // Keep pinned source bytes live through this owner's upload only. The
        // caller remains responsible for outstanding numerical kernel uses.
        try {
            if (completion.recorded() && !completion.ready()) completion.synchronize();
        } catch (...) {
        }
    }

    modeling::cuda::FactorizedInteractionProblem problem{};
    int device = -1;
    cudaStream_t upload_stream_handle = nullptr;
    memory::cuda::DeviceBuffer storage;
    memory::cuda::PinnedBuffer staging;
    memory::cuda::Event completion;
    std::size_t storage_bytes = 0;
    layout::FactorizedInteractionView device_view{};
};

MinimalDirectDeviceLayout::MinimalDirectDeviceLayout(
    const MinimalDirectHostLayout& host,
    modeling::cuda::FactorizedInteractionProblem problem,
    int device, const memory::cuda::Stream& upload_stream)
    : state_(std::make_unique<State>(host, problem, device, upload_stream)) {}

MinimalDirectDeviceLayout::~MinimalDirectDeviceLayout() noexcept = default;
MinimalDirectDeviceLayout::MinimalDirectDeviceLayout(
    MinimalDirectDeviceLayout&&) noexcept = default;
MinimalDirectDeviceLayout& MinimalDirectDeviceLayout::operator=(
    MinimalDirectDeviceLayout&&) noexcept = default;

const layout::FactorizedInteractionView& MinimalDirectDeviceLayout::view() const noexcept {
    static const layout::FactorizedInteractionView empty{};
    return state_ ? state_->device_view : empty;
}

bool MinimalDirectDeviceLayout::accepts(
    modeling::cuda::FactorizedInteractionProblem problem) const noexcept {
    return state_ && state_->problem.cells == problem.cells &&
           state_->problem.channels == problem.channels;
}

int MinimalDirectDeviceLayout::device() const noexcept {
    return state_ ? state_->device : -1;
}

std::size_t MinimalDirectDeviceLayout::bytes() const noexcept {
    return state_ ? state_->storage_bytes : 0;
}

void MinimalDirectDeviceLayout::wait_on(const memory::cuda::Stream& stream) const {
    if (!state_)
        throw std::logic_error("minimal direct device layout is moved from");
    if (stream.device() != state_->device)
        throw std::invalid_argument("direct control wait stream/device mismatch");
    if (stream.get() != state_->upload_stream_handle)
        stream.wait(state_->completion);
}

} // namespace cellerator::bench::factorized_interactions::controls
