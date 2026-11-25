module;

#include <stddef.h>
#include <stdint.h>
#include <Cellerator/memory/view.hh>
#include <Cellerator/tensor/base/compressed.hh>
#include "dgcmatrix_rds.hh"
#include "rds_index.hh"

module cellerator.rds;

namespace cellerator::rds::detail {

struct Transfer {
    const RdsFile *owner = nullptr;
    const IndexedFile *file = nullptr;
    uint64_t generation = 0;
    GzipCursor indices, values;
    GzipCursor *names = nullptr;
    bool names_enabled = false;
    unsigned char discard[gzip_io_bytes];

    ~Transfer() noexcept { delete names; }
};

} // namespace cellerator::rds::detail

namespace cellerator::rds {
namespace {

using detail::GzipCursor;
using detail::IndexedFile;
using detail::IndexStatus;
using detail::Transfer;

static_assert(sizeof(double) == 8 && __DBL_MANT_DIG__ == 53 && __DBL_MAX_EXP__ == 1024);

Status public_status(IndexStatus status) noexcept {
    switch (status) {
    case IndexStatus::ok: return Status::ok;
    case IndexStatus::invalid_argument: return Status::invalid_argument;
    case IndexStatus::io_error: return Status::io_error;
    case IndexStatus::unsupported: return Status::unsupported;
    case IndexStatus::invalid_format: return Status::invalid_format;
    case IndexStatus::invalid_index: return Status::invalid_index;
    case IndexStatus::stale_index: return Status::stale_index;
    }
    return Status::io_error;
}

bool host(memory::placement where) noexcept {
    using memory::domain;
    return where.kind == domain::host || where.kind == domain::host_numa ||
           where.kind == domain::host_pinned || where.kind == domain::host_pinned_write_combined;
}

struct Span {
    uintptr_t start = 0, end = 0;
};

template<class T>
bool output_span(memory::array_view<T> view, size_t used, Span& span) noexcept {
    if (!used) return true;
    uintptr_t address = reinterpret_cast<uintptr_t>(view.data);
    if (!host(view.where) || !address || address % alignof(T) ||
        used > (UINTPTR_MAX - address)/sizeof(T)) return false;
    span = {address, address + used*sizeof(T)};
    return true;
}

bool overlaps(Span a, Span b) noexcept {
    return a.start < b.end && b.start < a.end;
}

bool range_fits(ColumnRange range, uint64_t extent) noexcept {
    return range.start <= extent && range.count <= extent - range.start;
}

bool run_before(const ReadRun& a, const ReadRun& b) noexcept {
    return a.source.start < b.source.start;
}

void sift_run(ReadRun *runs, size_t root, size_t count) noexcept {
    const ReadRun run = runs[root];
    while (root < count/2) {
        size_t child = 2*root + 1;
        if (child+1 < count && run_before(runs[child], runs[child+1])) ++child;
        if (!run_before(run, runs[child])) break;
        runs[root] = runs[child];
        root = child;
    }
    runs[root] = run;
}

void sort_runs(ReadRun *runs, size_t count) noexcept {
    // The heap uses the supplied run array and needs no auxiliary stack.
    for (size_t root = count/2; root;) sift_run(runs, --root, count);
    for (size_t end = count; end > 1;) {
        const ReadRun last = runs[--end];
        runs[end] = runs[0];
        runs[0] = last;
        sift_run(runs, 0, end);
    }
}

bool load_range(GzipCursor& cursor, const IndexedFile& file, uint64_t at,
                void *output, size_t bytes) noexcept {
    return cursor.read_at(detail::checkpoint_at(file, at), at, output, bytes);
}

template<class T, class Bits>
void native_endian(T *data, size_t count) noexcept {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    for (size_t i = 0; i < count; ++i) {
        Bits value;
        __builtin_memcpy(&value, data + i, sizeof(value));
        if constexpr (sizeof(Bits) == 4) value = __builtin_bswap32(value);
        else value = __builtin_bswap64(value);
        __builtin_memcpy(data + i, &value, sizeof(value));
    }
#else
    (void)data;
    (void)count;
#endif
}

void write_offsets(const int32_t *source, uint32_t count, size_t output_first, int32_t *offsets) noexcept {
    size_t first = size_t(source[0]);
    for (uint32_t col = 0; col < count; ++col)
        offsets[col + 1] = int32_t(output_first + size_t(source[col + 1]) - first);
}

Status read_columns(Transfer& io, ColumnRange range, size_t output_first, double *values,
                    int32_t *indices, int32_t *offsets) noexcept {
    const auto& file = *io.file;
    const int32_t *source = file.offsets + range.start;
    size_t first = size_t(source[0]);
    size_t entries = size_t(source[range.count]) - first;
    write_offsets(source, range.count, output_first, offsets);
    if (!entries) return Status::ok;

    const auto& h = *file.header;
    indices += output_first;
    values += output_first;
    if (!load_range(io.indices, file, h.row_indices + first*sizeof(int32_t),
                    indices, entries*sizeof(int32_t))) return Status::io_error;
    native_endian<int32_t, uint32_t>(indices, entries);
    size_t at = 0;
    for (uint32_t col = 0; col < range.count; ++col) {
        size_t end = size_t(source[col+1]) - first;
        int32_t previous = -1;
        for (; at < end; ++at) {
            int32_t row = indices[at];
            if (row <= previous || uint64_t(row) >= h.rows) return Status::invalid_format;
            previous = row;
        }
    }
    if (!load_range(io.values, file, h.values + first*sizeof(double),
                    values, entries*sizeof(double))) return Status::io_error;
    native_endian<double, uint64_t>(values, entries);
    return Status::ok;
}

Status read_ordered(Transfer& io, const ReadRun *runs, size_t count, double *values,
                    int32_t *indices, int32_t *offsets) noexcept {
    const auto *p = io.file->offsets;
    uint32_t covered = 0;
    ReadRun anchor{};
    for (size_t r = 0; r < count; ++r) {
        const auto& run = runs[r];
        auto range = run.source;
        uint32_t end = range.start + range.count;
        size_t column = run.output_column, entry = run.output_entry;
        // Source order keeps the shared prefix inside the previously decoded
        // run with the furthest end. Its output occupies a disjoint interval.
        if (range.start < covered) {
            uint32_t shared_end = end < covered ? end : covered;
            uint32_t shared = shared_end - range.start;
            size_t entries = size_t(p[shared_end] - p[range.start]);
            if (entries) {
                size_t from = anchor.output_entry + size_t(p[range.start] - p[anchor.source.start]);
                __builtin_memcpy(values + entry, values + from, entries*sizeof(*values));
                __builtin_memcpy(indices + entry, indices + from, entries*sizeof(*indices));
            }
            write_offsets(p + range.start, shared, entry, offsets + column);
            column += shared;
            entry += entries;
            range = {shared_end, end - shared_end};
        }
        if (!range.count) continue;
        auto status = read_columns(io, range, entry, values, indices, offsets + column);
        if (status != Status::ok) return status;
        anchor = run;
        covered = end;
    }
    return Status::ok;
}

struct NameInput {
    GzipCursor& cursor;
    const detail::GzipCheckpoint& checkpoint;
    uint64_t position, limit;

    static bool take(void *opaque, void *out, size_t bytes) noexcept {
        auto& in = *static_cast<NameInput*>(opaque);
        if (bytes > in.limit - in.position) return false;
        if (out && !in.cursor.read_at(in.checkpoint, in.position, out, bytes)) return false;
        in.position += bytes;
        return true;
    }
};

Status copy_names(Transfer& io, unsigned axis, ColumnRange range, bool sizing,
                  AxisName *names, char *bytes, size_t capacity, size_t& needed) noexcept {
    const auto& file = *io.file;
    constexpr uint32_t stride = detail::name_checkpoint_stride;
    uint64_t start = file.names[axis][range.start/stride];
    NameInput reader{*io.names, detail::checkpoint_at(file, start), start, file.header->decoded};
    detail::XdrInput xdr{&reader, NameInput::take, start};
    uint32_t skip = range.start%stride;
    uint32_t count = skip + range.count;
    size_t used = 0;
    for (uint32_t i = 0; i < count; ++i) {
        detail::XdrStringHeader name{};
        auto status = detail::read_xdr_string_header(xdr, name);
        if (status == detail::XdrStatus::io_error) return Status::io_error;
        if (status != detail::XdrStatus::ok) return Status::invalid_format;
        if (name.length < 0) {
            if (i >= skip && !sizing) names[i-skip] = {};
            continue;
        }
        size_t length = size_t(name.length);
        if (i < skip) {
            if (detail::read_xdr_bytes(xdr, nullptr, length) != detail::XdrStatus::ok)
                return Status::io_error;
            continue;
        }
        if (length >= SIZE_MAX - used) return Status::invalid_format;
        size_t next = used + length + 1;
        char *text = !sizing && next <= capacity ? bytes + used : nullptr;
        if (detail::read_xdr_bytes(xdr, text, length) != detail::XdrStatus::ok)
            return Status::io_error;
        if (text) {
            text[length] = '\0';
            names[i-skip] = {text, length, static_cast<NameEncoding>(name.encoding)};
        }
        used = next;
    }
    needed = used;
    return !sizing && used > capacity ? Status::insufficient_capacity : Status::ok;
}

} // namespace

Status build_index(const char *source, const char *index_path, IoStats& stats) noexcept {
    return build_index(source, index_path, 8*1024*1024, stats);
}

Status build_index(const char *source, const char *index_path,
                   uint64_t checkpoint_spacing_bytes, IoStats& stats) noexcept {
    detail::GzipStats gzip{};
    auto status = detail::create_index(source, index_path, checkpoint_spacing_bytes, gzip);
    stats = {gzip.compressed_bytes, gzip.decoded_bytes};
    return public_status(status);
}

RdsFile::~RdsFile() noexcept { close(); }

Status RdsFile::open(const char *source, const char *index_path) noexcept {
    IndexedFile *next = nullptr;
    auto status = detail::open_index(source, index_path, next);
    if (status == IndexStatus::ok) {
        close();
        file_ = next;
        ++generation_;
    }
    return public_status(status);
}

void RdsFile::close() noexcept {
    detail::close_index(file_);
    file_ = nullptr;
}

FileInfo RdsFile::info() const noexcept {
    if (!file_) return {};
    const auto& h = *file_->header;
    return {uint32_t(h.rows), uint32_t(h.columns), uint32_t(h.entries),
            {h.name_count[0] != 0, h.name_count[1] != 0}, h.source.bytes, h.index_bytes, h.decoded};
}

Status RdsFile::prepare_read(const ColumnRange *ranges, size_t count, PreparedRead& read) const noexcept {
    if (!file_ || (!ranges && count)) return Status::invalid_argument;
    const auto& file = *file_;
    ReadRequirements result{size_t(file.header->rows), 0, 0, 1};
    for (size_t i = 0; i < count; ++i) {
        auto range = ranges[i];
        if (!range_fits(range, file.header->columns)) return Status::invalid_argument;
        size_t entries = size_t(file.offsets[range.start+range.count] - file.offsets[range.start]);
        if (range.count > SIZE_MAX - result.offsets || entries > INT32_MAX - result.entries)
            return Status::invalid_argument;
        result.columns += range.count;
        result.offsets += range.count;
        result.entries += entries;
    }
    read.file_ = this;
    read.generation_ = generation_;
    read.ranges_ = ranges;
    read.runs_ = nullptr;
    read.count_ = count;
    read.required_ = result;
    return Status::ok;
}

Status RdsFile::prepare_read(const ColumnRange *ranges, size_t count,
                             memory::array_view<ReadRun> workspace, PreparedRead& read) const noexcept {
    if (!file_) return Status::invalid_argument;
    if (workspace.count < count) return Status::insufficient_capacity;
    Span source, target;
    constexpr memory::placement local{memory::domain::host};
    if (!output_span(memory::array_view<const ColumnRange>{ranges, count, local}, count, source) ||
        !output_span(workspace, count, target) || overlaps(source, target)) return Status::invalid_argument;
    PreparedRead next;
    auto status = prepare_read(ranges, count, next);
    if (status != Status::ok) return status;

    const auto *p = file_->offsets;
    size_t columns = 0, entries = 0, used = 0;
    bool sorted = true;
    auto *runs = workspace.data;
    for (size_t i = 0; i < count; ++i) {
        auto range = ranges[i];
        if (!range.count) continue;
        if (used && runs[used-1].source.start + runs[used-1].source.count == range.start) {
            runs[used-1].source.count += range.count;
        } else {
            runs[used] = {range, columns, entries};
            if (used && sorted) sorted = !run_before(runs[used], runs[used-1]);
            ++used;
        }
        columns += range.count;
        entries += size_t(p[range.start + range.count] - p[range.start]);
    }
    if (!sorted) sort_runs(runs, used);
    next.ranges_ = nullptr;
    next.runs_ = used ? runs : nullptr;
    next.count_ = used;
    read = next;
    return Status::ok;
}

ReadContext::~ReadContext() noexcept { delete transfer_; }

Status ReadContext::prepare_transfer(const RdsFile& file) noexcept {
    return prepare_transfer(file, TransferMode::numeric_and_names);
}

Status ReadContext::prepare_transfer(const RdsFile& file, TransferMode mode) noexcept {
    if (!file.file_ || (mode != TransferMode::numeric && mode != TransferMode::numeric_and_names))
        return Status::invalid_argument;
    auto *next = transfer_;
    const bool same_session = next && next->owner == &file && next->generation == file.generation_;
    const auto& header = *file.file_->header;
    const bool names_enabled = mode == TransferMode::numeric_and_names &&
                               (header.name_count[0] || header.name_count[1]);
    if (same_session && next->names_enabled == names_enabled) return Status::ok;

    int fd = file.file_->fd;
    try {
        if (!next) {
            next = new Transfer;
            if (!next->indices.initialize(fd, next->discard) ||
                !next->values.initialize(fd, next->discard)) {
                delete next;
                return Status::io_error;
            }
        }
        if (names_enabled && !next->names) {
            auto *names = new GzipCursor;
            if (!names->initialize(fd, next->discard)) {
                delete names;
                if (next != transfer_) delete next;
                return Status::io_error;
            }
            next->names = names;
        }
    } catch (...) {
        if (next != transfer_) delete next;
        return Status::io_error;
    }

    if (!same_session) {
        next->indices.rebind(fd);
        next->values.rebind(fd);
        if (next->names) next->names->rebind(fd);
        next->owner = &file;
        next->file = file.file_;
        next->generation = file.generation_;
    }
    next->names_enabled = names_enabled;
    transfer_ = next;
    return Status::ok;
}

Status ReadContext::read_into(const PreparedRead& read, memory::array_view<double> values,
                              memory::array_view<int32_t> indices, memory::array_view<int32_t> offsets,
                              CscView& output) noexcept {
    if (!transfer_ || read.file_ != transfer_->owner || read.generation_ != transfer_->generation ||
        transfer_->owner->file_ != transfer_->file ||
        transfer_->owner->generation_ != transfer_->generation) return Status::invalid_argument;
    const auto& req = read.required_;
    if (values.count < req.entries || indices.count < req.entries || offsets.count < req.offsets)
        return Status::insufficient_capacity;
    Span v, i, o;
    if (!output_span(values, req.entries, v) || !output_span(indices, req.entries, i) ||
        !output_span(offsets, req.offsets, o) || overlaps(v, i) || overlaps(v, o) || overlaps(i, o))
        return Status::invalid_argument;

    if (read.runs_) {
        auto start = reinterpret_cast<uintptr_t>(read.runs_);
        Span plan{start, start + read.count_*sizeof(ReadRun)};
        if (overlaps(plan, v) || overlaps(plan, i) || overlaps(plan, o)) return Status::invalid_argument;
        offsets.data[0] = 0;
        auto status = read_ordered(*transfer_, read.runs_, read.count_, values.data, indices.data, offsets.data);
        if (status != Status::ok) return status;
    } else {
        size_t out_col = 0;
        offsets.data[0] = 0;
        for (size_t n = 0; n < read.count_;) {
            auto range = read.ranges_[n++];
            if (!range.count) continue;
            while (n < read.count_ && read.ranges_[n].start == range.start + range.count)
                range.count += read.ranges_[n++].count;
            auto status = read_columns(*transfer_, range, size_t(offsets.data[out_col]),
                                       values.data, indices.data, offsets.data + out_col);
            if (status != Status::ok) return status;
            out_col += range.count;
        }
    }
    output = {{req.rows, req.columns, req.entries, tensor::base::CompressedAxis::column,
               offsets.data, indices.data}, values.data};
    return Status::ok;
}

Status ReadContext::read_names_into(Axis axis, ColumnRange range, memory::array_view<AxisName> names,
                                    memory::array_view<char> bytes, size_t& needed) noexcept {
    needed = 0;
    if (!transfer_ || transfer_->owner->file_ != transfer_->file ||
        transfer_->owner->generation_ != transfer_->generation ||
        (axis != Axis::features && axis != Axis::cells)) return Status::invalid_argument;
    unsigned selected = axis == Axis::features ? 0 : 1;
    const auto& h = *transfer_->file->header;
    if (!range_fits(range, selected ? h.columns : h.rows)) return Status::invalid_argument;
    if (!range.count) return Status::ok;
    if (!transfer_->names_enabled || !h.name_count[selected]) return Status::unsupported;

    bool sizing = !names.data && !names.count && !bytes.data && !bytes.count;
    if (!sizing && names.count < range.count) return Status::insufficient_capacity;
    Span n, b;
    if (!sizing && (!output_span(names, range.count, n) || !output_span(bytes, bytes.count, b) ||
                    overlaps(n, b))) return Status::invalid_argument;
    return copy_names(*transfer_, selected, range, sizing, names.data, bytes.data, bytes.count, needed);
}

IoStats ReadContext::stats() const noexcept {
    if (!transfer_) return {};
    auto i = transfer_->indices.stats(), x = transfer_->values.stats();
    auto n = transfer_->names ? transfer_->names->stats() : detail::GzipStats{};
    return {i.compressed_bytes + x.compressed_bytes + n.compressed_bytes,
            i.decoded_bytes + x.decoded_bytes + n.decoded_bytes};
}

} // namespace cellerator::rds
