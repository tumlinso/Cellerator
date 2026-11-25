#pragma once

#include <Cellerator/execution/streaming/fetch.hh>

namespace cellerator::test {

// Exercise the adapter, queue, and SELL packet together against known sparse rows.
template<class V, class I, class O>
bool check_source_packet(execution::streaming::FetchSource source,
                         execution::streaming::CellRange cells,
                         const V *values, const I *indices, const O *offsets) {
    using namespace execution::streaming;
    using memory::status;
    if (cells.count > 4 || offsets[cells.count] > 16) return false;

    V scratch_values[16]; I scratch_indices[16]; O scratch_offsets[5];
    FetchRun run{};
    unsigned lengths[4], order[4];
    uint64_t source_lengths[4];
    layout::SlicedEllRow logical[4], map[4];
    layout::SlicedEllSlice slices[2];
    alignas(tensor::base::device::SlicedEllLayout::alignment) unsigned char packet[512];
    const FetchWorkspace workspace{
        {scratch_values, scratch_indices, scratch_offsets,
         sizeof(scratch_values), sizeof(scratch_indices), sizeof(scratch_offsets)},
        &run, 1, lengths, order, source_lengths, logical, 4, slices, 2};
    const FetchSlotStorage storage{{}, packet, sizeof(packet)};
    const FetchWorker inline_worker{nullptr, [](void *, FetchJob job, void *context) noexcept {
        job(context);
        return true;
    }};
    const FetchOptions options{FetchFormat::sliced_ell, 1, {2, 4, true, true}};
    FetchQueue queue;
    FetchTicket ticket;
    if (queue.prepare(source, inline_worker, {}, options, workspace, &storage, 1) != status::success ||
        queue.fetch({&cells, 1, 3, 19, map, 4}, &ticket) != status::success) return false;
    const auto report = queue.poll(ticket);
    tensor::base::device::SlicedEllView view{};
    bool ok = report.state == FetchState::ready && report.batch.batch == 3 &&
        report.batch.source.feature_axis == source.info.feature_axis &&
        report.batch.packet_layout.bind(packet, sizeof(packet), &view) == status::success &&
        view.rows == cells.count && view.nnz == unsigned(offsets[cells.count]);
    unsigned seen = 0;
    for (unsigned slice = 0; ok && slice < view.slice_count; ++slice) {
        const unsigned width = view.slice_widths[slice];
        const unsigned first = view.slice_row_offsets[slice];
        for (unsigned row = first; ok && row < view.slice_row_offsets[slice+1]; ++row) {
            const uint64_t logical_row = map[row].logical_position - 19;
            if (logical_row >= cells.count || map[row].source_cell != cells.first + logical_row ||
                (seen & (1u << logical_row))) { ok = false; break; }
            seen |= 1u << logical_row;
            const auto begin = offsets[logical_row], end = offsets[logical_row+1];
            if (width < unsigned(end-begin)) { ok = false; break; }
            for (unsigned i = 0; i < width; ++i) {
                const size_t slot = view.slice_slot_offsets[slice] + size_t(row-first)*width + i;
                const bool valid = i < unsigned(end-begin);
                uint16_t bits;
                __builtin_memcpy(&bits, view.val + slot, sizeof(bits));
                const unsigned column = valid ? unsigned(indices[begin+i]) :
                    unsigned(tensor::base::sliced_ell_invalid_col);
                const uint16_t expected = valid ? real::round_f16(values[begin+i]) : 0;
                if (view.col_idx[slot] != column || bits != expected) { ok = false; break; }
            }
        }
    }
    return queue.release(ticket) == status::success && queue.drain() && ok &&
           seen == (1u << cells.count) - 1;
}

} // namespace cellerator::test
