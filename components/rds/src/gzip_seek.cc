#include "gzip_seek.hh"

#include <errno.h>
#include <limits.h>
#include <unistd.h>

namespace cellerator::rds::detail {
namespace {

ssize_t read_bytes(int fd, uint64_t offset, unsigned char *out, size_t count) noexcept {
    static_assert(sizeof(off_t) >= sizeof(int64_t));
    if (offset > uint64_t(INT64_MAX)) return -1;
    ssize_t got;
    do { got = pread(fd, out, count, off_t(offset)); } while (got < 0 && errno == EINTR);
    return got;
}

bool refill(z_stream &stream, int fd, uint64_t &end, unsigned char *input,
            GzipStats &stats) noexcept {
    if (stream.avail_in) return true;
    const ssize_t got = read_bytes(fd, end, input, gzip_io_bytes);
    if (got <= 0) return false;
    end += uint64_t(got);
    stats.compressed_bytes += uint64_t(got);
    stream.next_in = input;
    stream.avail_in = uInt(got);
    return true;
}

} // namespace

GzipScan::~GzipScan() noexcept {
    if (stream_.state) inflateEnd(&stream_);
}

bool GzipScan::initialize(int fd, uint64_t checkpoint_spacing_bytes, EmitGzipCheckpoint emit, void *context) noexcept {
    if (fd < 0 || !emit || stream_.state || inflateInit2(&stream_, MAX_WBITS + 16) != Z_OK)
        return false;
    fd_ = fd;
    checkpoint_spacing_bytes_ = checkpoint_spacing_bytes;
    emit_ = emit;
    context_ = context;
    return true;
}

bool GzipScan::fill() noexcept {
    next_ = size_ = 0;
    while (!size_ && !ended_) {
        if (!refill(stream_, fd_, input_end_, input_, stats_)) return false;
        stream_.next_out = output_;
        stream_.avail_out = gzip_io_bytes;
        const int status = inflate(&stream_, Z_BLOCK);
        if (status != Z_OK && status != Z_STREAM_END) return false;
        size_ = gzip_io_bytes - stream_.avail_out;
        stats_.decoded_bytes += size_;

        // At a block boundary, data_type exposes the held bits in the last consumed byte.
        if ((stream_.data_type & 192) == 128 &&
            (!indexed_ || stats_.decoded_bytes - last_checkpoint_ >= checkpoint_spacing_bytes_)) {
            GzipCheckpoint point{};
            uInt history_size = 0;
            if (inflateGetDictionary(&stream_, point.history, &history_size) != Z_OK) return false;
            point.decoded = stats_.decoded_bytes;
            point.compressed = input_end_ - stream_.avail_in;
            point.bits = uint32_t(stream_.data_type & 7);
            point.history_size = uint32_t(history_size);
            if (!emit_(context_, point)) return false;
            last_checkpoint_ = point.decoded;
            indexed_ = true;
        }
        if (status == Z_STREAM_END) {
            unsigned char trailer;
            if (!indexed_ || stream_.avail_in || read_bytes(fd_, input_end_, &trailer, 1) != 0)
                return false;
            ended_ = true;
        }
    }
    return true;
}

bool GzipScan::read(void *out, size_t count, size_t &got) noexcept {
    got = 0;
    if (fd_ < 0 || failed_) return false;
    auto *dst = static_cast<unsigned char*>(out);
    while (got < count) {
        if (next_ == size_ && !fill()) { failed_ = true; return false; }
        if (next_ == size_) break;
        const size_t available = size_ - next_;
        const size_t take = count - got < available ? count - got : available;
        if (dst) __builtin_memcpy(dst + got, output_ + next_, take);
        next_ += take;
        got += take;
    }
    return true;
}

bool GzipScan::finish() noexcept {
    size_t got;
    return read(nullptr, 1, got) && !got;
}

GzipCursor::~GzipCursor() noexcept {
    if (stream_.state) inflateEnd(&stream_);
}

bool GzipCursor::initialize(int fd, unsigned char (&discard)[gzip_io_bytes]) noexcept {
    if (fd < 0 || stream_.state || inflateInit2(&stream_, -MAX_WBITS) != Z_OK) return false;
    // A one-byte dictionary provisions the full zlib window. The first restart resets it.
    const unsigned char seed = 0;
    if (inflateSetDictionary(&stream_, &seed, 1) != Z_OK) return false;
    fd_ = fd;
    discard_ = discard;
    return true;
}

bool GzipCursor::restart(const GzipCheckpoint &point) noexcept {
    if (inflateReset(&stream_) != Z_OK) return false;
    stream_.avail_in = 0;
    if (point.bits) {
        unsigned char previous;
        if (read_bytes(fd_, point.compressed - 1, &previous, 1) != 1) return false;
        ++stats_.compressed_bytes;
        if (inflatePrime(&stream_, int(point.bits), previous >> (8 - point.bits)) != Z_OK)
            return false;
    }
    if (point.history_size &&
        inflateSetDictionary(&stream_, point.history, point.history_size) != Z_OK) return false;
    input_end_ = point.compressed;
    decoded_ = point.decoded;
    ended_ = false;
    return true;
}

bool GzipCursor::decode(unsigned char *out, size_t count) noexcept {
    while (count) {
        if (ended_ || !refill(stream_, fd_, input_end_, input_, stats_)) return false;
        const uInt capacity = count > UINT_MAX ? UINT_MAX : uInt(count);
        stream_.next_out = out;
        stream_.avail_out = capacity;
        const int status = inflate(&stream_, Z_NO_FLUSH);
        if (status != Z_OK && status != Z_STREAM_END) return false;
        const uInt produced = capacity - stream_.avail_out;
        stats_.decoded_bytes += produced;
        decoded_ += produced;
        out += produced;
        count -= produced;
        ended_ = status == Z_STREAM_END;
    }
    return true;
}

bool GzipCursor::read_at(const GzipCheckpoint &point, uint64_t offset,
                         void *out, size_t count) noexcept {
    if (fd_ < 0 || point.decoded > offset || point.compressed > uint64_t(INT64_MAX) ||
        point.bits > 7 || (point.bits && !point.compressed) ||
        point.history_size > gzip_history_bytes || count > UINT64_MAX - offset || (count && !out))
        return false;
    if (!count) return true;
    if (!started_ || offset < decoded_ || decoded_ < point.decoded) {
        started_ = restart(point);
        if (!started_) return false;
    }
    while (decoded_ < offset) {
        const uint64_t skip = offset - decoded_;
        if (!decode(discard_, skip < gzip_io_bytes ? size_t(skip) : gzip_io_bytes)) {
            started_ = false;
            return false;
        }
    }
    started_ = decode(static_cast<unsigned char*>(out), count);
    return started_;
}

} // namespace cellerator::rds::detail
