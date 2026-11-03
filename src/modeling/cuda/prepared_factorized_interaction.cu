#include <Cellerator/modeling/cuda/factorized_interaction.hh>

#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace cellerator::modeling::cuda {
namespace {

struct Region {
    std::size_t offset = 0;
    std::size_t bytes = 0;
};

std::size_t checked_add(std::size_t left, std::size_t right) {
    if (right > std::numeric_limits<std::size_t>::max() - left)
        throw std::overflow_error("factorized device layout size overflow");
    return left + right;
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    const auto remainder = value % alignment;
    return remainder == 0 ? value : checked_add(value, alignment - remainder);
}

template<class T>
Region append_region(std::size_t& total, const std::vector<T>& values) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "device layout records must be trivially copyable");
    if (values.empty()) return {};
    if (values.size() > std::numeric_limits<std::size_t>::max() / sizeof(T))
        throw std::overflow_error("factorized device layout array size overflow");
    total = align_up(total, alignof(T));
    Region result{total, values.size() * sizeof(T)};
    total = checked_add(total, result.bytes);
    return result;
}

template<class T>
void stage_region(void* staging, const std::vector<T>& values, Region region) {
    if (region.bytes)
        std::memcpy(static_cast<char*>(staging) + region.offset,
                    values.data(), region.bytes);
}

template<class T>
const T* device_region(const void* storage, Region region) noexcept {
    return region.bytes
        ? reinterpret_cast<const T*>(static_cast<const char*>(storage) + region.offset)
        : nullptr;
}

struct PostingRegions {
    Region offsets;
    Region entries;
};

PostingRegions append_posting_regions(
    std::size_t& total, const layout::FactorizedInteractionPostings& postings) {
    return {append_region(total, postings.offsets),
            append_region(total, postings.entries)};
}

void stage_posting_regions(void* staging,
                           const layout::FactorizedInteractionPostings& postings,
                           PostingRegions regions) {
    stage_region(staging, postings.offsets, regions.offsets);
    stage_region(staging, postings.entries, regions.entries);
}

layout::FactorizedInteractionPostingsView device_postings(
    const void* storage, PostingRegions regions) noexcept {
    return {device_region<std::uint64_t>(storage, regions.offsets),
            device_region<std::uint64_t>(storage, regions.entries)};
}

struct Regions {
    Region mechanisms, terms, packets, packet_mechanisms, mechanism_packets;
    Region packet_inputs, packet_contexts, packet_outputs, role_refs, context_refs;
    PostingRegions local_input_roles, local_output_mechanisms, local_context_terms;
    PostingRegions input_packets, context_packets, output_packets;
    PostingRegions alpha_mechanisms, bias_mechanisms, gamma_terms, context_terms;
};

Regions append_regions(std::size_t& total,
                       const layout::FactorizedInteractionLayout& host) {
    Regions r;
    r.mechanisms = append_region(total, host.mechanisms);
    r.terms = append_region(total, host.terms);
    r.packets = append_region(total, host.packets);
    r.packet_mechanisms = append_region(total, host.packet_mechanisms);
    r.mechanism_packets = append_region(total, host.mechanism_packets);
    r.packet_inputs = append_region(total, host.packet_inputs);
    r.packet_contexts = append_region(total, host.packet_contexts);
    r.packet_outputs = append_region(total, host.packet_outputs);
    r.role_refs = append_region(total, host.role_refs);
    r.context_refs = append_region(total, host.context_refs);
    r.local_input_roles = append_posting_regions(total, host.local_input_roles);
    r.local_output_mechanisms = append_posting_regions(total, host.local_output_mechanisms);
    r.local_context_terms = append_posting_regions(total, host.local_context_terms);
    r.input_packets = append_posting_regions(total, host.input_packets);
    r.context_packets = append_posting_regions(total, host.context_packets);
    r.output_packets = append_posting_regions(total, host.output_packets);
    r.alpha_mechanisms = append_posting_regions(total, host.alpha_mechanisms);
    r.bias_mechanisms = append_posting_regions(total, host.bias_mechanisms);
    r.gamma_terms = append_posting_regions(total, host.gamma_terms);
    r.context_terms = append_posting_regions(total, host.context_terms);
    return r;
}

void stage_regions(void* staging, const layout::FactorizedInteractionLayout& host,
                   const Regions& r) {
    stage_region(staging, host.mechanisms, r.mechanisms);
    stage_region(staging, host.terms, r.terms);
    stage_region(staging, host.packets, r.packets);
    stage_region(staging, host.packet_mechanisms, r.packet_mechanisms);
    stage_region(staging, host.mechanism_packets, r.mechanism_packets);
    stage_region(staging, host.packet_inputs, r.packet_inputs);
    stage_region(staging, host.packet_contexts, r.packet_contexts);
    stage_region(staging, host.packet_outputs, r.packet_outputs);
    stage_region(staging, host.role_refs, r.role_refs);
    stage_region(staging, host.context_refs, r.context_refs);
    stage_posting_regions(staging, host.local_input_roles, r.local_input_roles);
    stage_posting_regions(staging, host.local_output_mechanisms, r.local_output_mechanisms);
    stage_posting_regions(staging, host.local_context_terms, r.local_context_terms);
    stage_posting_regions(staging, host.input_packets, r.input_packets);
    stage_posting_regions(staging, host.context_packets, r.context_packets);
    stage_posting_regions(staging, host.output_packets, r.output_packets);
    stage_posting_regions(staging, host.alpha_mechanisms, r.alpha_mechanisms);
    stage_posting_regions(staging, host.bias_mechanisms, r.bias_mechanisms);
    stage_posting_regions(staging, host.gamma_terms, r.gamma_terms);
    stage_posting_regions(staging, host.context_terms, r.context_terms);
}

layout::FactorizedInteractionView make_device_view(
    const layout::FactorizedInteractionLayout& host, const void* storage,
    const Regions& r) noexcept {
    auto result = host.view();
    result.mechanisms = device_region<layout::FactorizedInteractionRecord>(storage, r.mechanisms);
    result.terms = device_region<layout::FactorizedInteractionTermRecord>(storage, r.terms);
    result.packets = device_region<layout::FactorizedInteractionPacket>(storage, r.packets);
    result.packet_mechanisms = device_region<std::uint64_t>(storage, r.packet_mechanisms);
    result.mechanism_packets = device_region<std::uint64_t>(storage, r.mechanism_packets);
    result.packet_inputs = device_region<std::uint64_t>(storage, r.packet_inputs);
    result.packet_contexts = device_region<std::uint64_t>(storage, r.packet_contexts);
    result.packet_outputs = device_region<std::uint64_t>(storage, r.packet_outputs);
    result.role_refs = device_region<std::uint8_t>(storage, r.role_refs);
    result.context_refs = device_region<std::uint8_t>(storage, r.context_refs);
    result.local_input_roles = device_postings(storage, r.local_input_roles);
    result.local_output_mechanisms = device_postings(storage, r.local_output_mechanisms);
    result.local_context_terms = device_postings(storage, r.local_context_terms);
    result.input_packets = device_postings(storage, r.input_packets);
    result.context_packets = device_postings(storage, r.context_packets);
    result.output_packets = device_postings(storage, r.output_packets);
    result.alpha_mechanisms = device_postings(storage, r.alpha_mechanisms);
    result.bias_mechanisms = device_postings(storage, r.bias_mechanisms);
    result.gamma_terms = device_postings(storage, r.gamma_terms);
    result.context_terms = device_postings(storage, r.context_terms);
    return result;
}

} // namespace

struct FactorizedInteractionDeviceLayout::State {
    State(const layout::FactorizedInteractionLayout& host, int selected_device,
          const memory::cuda::Stream& upload_stream)
        : device(selected_device), upload_handle(upload_stream.get()),
          completion(selected_device, false) {
        if (device < 0) throw std::invalid_argument("device must be nonnegative");
        if (upload_stream.device() != device)
            throw std::invalid_argument("factorized layout upload stream/device mismatch");
        layout::validate_factorized_interactions(host);

        const auto regions = append_regions(storage_bytes, host);
        storage = memory::cuda::DeviceBuffer(storage_bytes, device);
        staging = memory::cuda::PinnedBuffer(storage_bytes);
        if (storage_bytes) {
            stage_regions(staging.data(), host, regions);
            storage.upload_async(staging.data(), storage_bytes, upload_stream);
        }
        device_view = make_device_view(host, storage.data(), regions);
        completion.record(upload_stream);
    }

    ~State() noexcept {
        // The pinned staging bytes must remain live until their one bulk H2D
        // transfer completes. This drains only that structural upload; the
        // caller remains responsible for all queued numerical uses of view().
        try {
            if (completion.recorded() && !completion.ready()) completion.synchronize();
        } catch (...) {
        }
    }

    int device = -1;
    cudaStream_t upload_handle = nullptr;
    memory::cuda::DeviceBuffer storage;
    memory::cuda::PinnedBuffer staging;
    memory::cuda::Event completion;
    std::size_t storage_bytes = 0;
    layout::FactorizedInteractionView device_view{};
};

FactorizedInteractionDeviceLayout::FactorizedInteractionDeviceLayout(
    const layout::FactorizedInteractionLayout& host, int device,
    const memory::cuda::Stream& upload_stream)
    : state_(std::make_unique<State>(host, device, upload_stream)) {}

FactorizedInteractionDeviceLayout::~FactorizedInteractionDeviceLayout() noexcept = default;
FactorizedInteractionDeviceLayout::FactorizedInteractionDeviceLayout(
    FactorizedInteractionDeviceLayout&&) noexcept = default;
FactorizedInteractionDeviceLayout& FactorizedInteractionDeviceLayout::operator=(
    FactorizedInteractionDeviceLayout&&) noexcept = default;

const layout::FactorizedInteractionView&
FactorizedInteractionDeviceLayout::view() const noexcept {
    static const layout::FactorizedInteractionView empty{};
    return state_ ? state_->device_view : empty;
}

int FactorizedInteractionDeviceLayout::device() const noexcept {
    return state_ ? state_->device : -1;
}

std::size_t FactorizedInteractionDeviceLayout::bytes() const noexcept {
    return state_ ? state_->storage_bytes : 0;
}

void FactorizedInteractionDeviceLayout::wait_on(
    const memory::cuda::Stream& stream) const {
    if (!state_) throw std::logic_error("factorized device layout is moved from");
    if (stream.device() != state_->device)
        throw std::invalid_argument("factorized layout wait stream/device mismatch");
    if (stream.get() != state_->upload_handle) stream.wait(state_->completion);
}

} // namespace cellerator::modeling::cuda
