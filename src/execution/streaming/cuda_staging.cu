#include <Cellerator/execution/streaming/cuda_staging.cuh>

namespace cellerator::execution::streaming {

using memory::status;
using memory::cuda::Stream;

status CudaStaging::prepare(FetchQueue &queue, memory::StripedBuffer &arena, const Stream &copy,
                            const StageEvents (&events)[2], size_t minimum_early_bytes) noexcept {
    for (const auto &slot : slots_) if (slot.live) return status::invalid_state;
    constexpr auto alignment = tensor::base::device::SlicedEllLayout::alignment;
    if (arena.device() != copy.device()) return status::invalid_placement;
    if (arena.bytes() < alignment) return status::capacity_exceeded;
    for (const auto &slot : events) {
        if (!slot.ready || !slot.last_reader || bool(slot.early_begin) != bool(slot.early_end) ||
            bool(slot.deferred_begin) != bool(slot.deferred_end)) return status::invalid_argument;
    }
    const memory::cuda::Event *list[]{
        events[0].ready, events[0].last_reader, events[0].early_begin,
        events[0].early_end, events[0].deferred_begin, events[0].deferred_end,
        events[1].ready, events[1].last_reader, events[1].early_begin,
        events[1].early_end, events[1].deferred_begin, events[1].deferred_end
    };
    for (unsigned i = 0; i < sizeof(list)/sizeof(list[0]); ++i) {
        if (!list[i]) continue;
        if (!list[i]->get()) return status::invalid_argument;
        if (list[i]->device() != copy.device()) return status::invalid_placement;
        for (unsigned j = 0; j < i; ++j)
            if (list[j] && list[i]->get() == list[j]->get()) return status::invalid_argument;
    }
    queue_ = &queue; arena_ = &arena; copy_ = &copy;
    slots_[0].events = events[0]; slots_[1].events = events[1];
    capacity_ = arena.bytes()/alignment*alignment;
    minimum_early_ = minimum_early_bytes;
    high_next_ = false; failed_ = false;
    return status::success;
}

CudaStaging::Slot *CudaStaging::lookup(DeviceTicket ticket) noexcept {
    if (ticket.owner != this || ticket.slot >= 2 || !ticket.generation) return nullptr;
    auto &slot = slots_[ticket.slot];
    return slot.live && slot.generation == ticket.generation ? &slot : nullptr;
}

void CudaStaging::upload(Slot &slot, size_t offset, size_t bytes, bool deferred) {
    auto *begin = deferred ? slot.events.deferred_begin : slot.events.early_begin;
    auto *end = deferred ? slot.events.deferred_end : slot.events.early_end;
    if (begin) begin->record(*copy_);
    arena_->upload_async(static_cast<const unsigned char *>(slot.packet_data)+offset,
                         bytes, *copy_, slot.batch.placement.offset+offset);
    if (end) end->record(*copy_);
}

status CudaStaging::advance() noexcept {
    if (!queue_) return status::invalid_state;
    if (failed_) return status::cuda_failure;
    try {
        for (auto &slot : slots_) {
            if (!slot.live || !slot.ready_recorded) continue;
            if (slot.host.owner && slot.events.ready->ready()) {
                const auto released = queue_->release(slot.host);
                if (released != status::success) {
                    failed_ = true;
                    return released;
                }
                slot.host = {};
                slot.packet_data = nullptr;
            }
            if (slot.released && !slot.host.owner &&
                (!slot.last_reader_recorded || slot.events.last_reader->ready()))
                slot.live = false;
        }
        for (unsigned i = 0; i < 2; ++i) {
            auto &slot = slots_[i];
            if (!slot.live || slot.ready_recorded) continue;
            const auto &previous = slots_[1-i];
            if (previous.live && previous.generation == slot.previous_generation) {
                if (!previous.released || !previous.ready_recorded) continue;
                copy_->wait(*(previous.last_reader_recorded ? previous.events.last_reader : previous.events.ready));
            }
            const auto &place = slot.batch.placement;
            if (place.deferred_bytes)
                upload(slot, slot.high ? 0 : place.early_bytes, place.deferred_bytes, true);
            slot.events.ready->record(*copy_);
            slot.ready_recorded = true;
        }
    } catch (...) {
        failed_ = true;
        return status::cuda_failure;
    }
    return status::success;
}

status CudaStaging::stage(FetchTicket packet, DeviceTicket *ticket) noexcept {
    if (!ticket || packet.owner != queue_) return status::invalid_argument;
    if (generation_ == UINT64_MAX) return status::arithmetic_overflow;
    const auto progressed = advance();
    if (progressed != status::success) return progressed;
    unsigned free = 2;
    for (unsigned i = 0; i < 2; ++i) {
        const auto &slot = slots_[i];
        if (!slot.live) free = i;
        else if (slot.host.owner && slot.host.slot == packet.slot && slot.host.generation == packet.generation)
            return status::invalid_argument;
    }
    if (free == 2) return status::capacity_exceeded;
    const auto host = queue_->poll(packet);
    if (host.state == FetchState::failed || host.state == FetchState::stale) return host.status;
    if (host.state != FetchState::ready) return status::invalid_state;
    if (!host.batch.packet_data || !host.batch.packet_layout.bytes()) return status::invalid_argument;

    const auto &previous = slots_[1-free];
    const bool high = previous.live ? !previous.high : high_next_;
    const size_t bytes = host.batch.packet_layout.bytes();
    PacketPlacement place{};
    auto result = place_packet(capacity_, previous.live ? previous.packet_bytes : 0,
                               bytes, high, minimum_early_, &place);
    if (result != status::success) return result;
    DeviceBatch batch{};
    result = arena_->view(place.offset, host.batch.packet_layout, &batch.view);
    if (result != status::success) return result;
    batch.source = host.batch.source; batch.batch = host.batch.batch;
    batch.row_map = host.batch.row_map; batch.placement = place;
    auto &slot = slots_[free];
    const auto events = slot.events;
    batch.ready = events.ready;
    slot = {};
    slot.live = true; slot.high = high; slot.generation = ++generation_;
    slot.previous_generation = place.deferred_bytes && previous.live ? previous.generation : 0;
    slot.host = packet; slot.events = events; slot.batch = batch;
    slot.packet_data = host.batch.packet_data; slot.packet_bytes = bytes;
    high_next_ = !high;
    *ticket = {this, slot.generation, free};
    // After admission the stager retains the host ticket even if submission
    // fails, since a prefix copy may already have been queued.
    try {
        if (place.early_bytes) upload(slot, place.early_offset, place.early_bytes, false);
    } catch (...) {
        failed_ = true;
        return status::cuda_failure;
    }
    return advance();
}

StageReport CudaStaging::poll(DeviceTicket ticket) noexcept {
    const auto result = advance();
    auto *slot = lookup(ticket);
    if (!slot || slot->released) return {};
    if (result != status::success) return {FetchState::failed, result, {}};
    if (!slot->ready_recorded) return {FetchState::pending, status::success, {}};
    slot->delivered = true;
    return {FetchState::ready, status::success, slot->batch};
}

status CudaStaging::release(DeviceTicket ticket, const Stream *last_reader) noexcept {
    auto *slot = lookup(ticket);
    if (!slot || slot->released) return status::invalid_argument;
    if (last_reader && !slot->ready_recorded) return status::invalid_state;
    if (last_reader && last_reader->device() != copy_->device()) return status::invalid_placement;
    try {
        if (last_reader) {
            slot->events.last_reader->record(*last_reader);
            slot->last_reader_recorded = true;
        }
        slot->released = true;
    } catch (...) {
        failed_ = true;
        return status::cuda_failure;
    }
    return advance();
}

status CudaStaging::drain() noexcept {
    if (!queue_) return status::invalid_state;
    for (const auto &slot : slots_) if (slot.live && slot.delivered && !slot.released) return status::invalid_state;
    try {
        copy_->synchronize();
        for (auto &slot : slots_) {
            if (!slot.live) continue;
            if (slot.last_reader_recorded) slot.events.last_reader->synchronize();
            if (slot.host.owner) {
                const auto released = queue_->release(slot.host);
                if (released != status::success) return released;
                slot.host = {};
            }
            slot.live = false;
        }
    } catch (...) {
        return status::cuda_failure;
    }
    queue_ = nullptr; arena_ = nullptr; copy_ = nullptr;
    return status::success;
}

} // namespace cellerator::execution::streaming
