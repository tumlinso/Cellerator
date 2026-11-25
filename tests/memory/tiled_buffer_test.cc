#include <Cellerator/memory/tiled/TiledBuffer.hh>

#include <cuda_fp16.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace {

namespace memory = cellerator::memory;
namespace gpu = memory::cuda;
using Tile = memory::Tile<__half>;
using Buffer = memory::TiledBuffer<__half>;

static_assert(sizeof(Tile) == 512 && alignof(Tile) == 32);
static_assert(!std::is_copy_constructible_v<Buffer> && !std::is_copy_assignable_v<Buffer>);
static_assert(std::is_nothrow_move_constructible_v<Buffer> && std::is_nothrow_move_assignable_v<Buffer>);

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Error, class F>
void require_error(F run, const char *message) {
    try { run(); }
    catch (const Error &) { return; }
    throw std::runtime_error(message);
}

void check_empty() {
    Buffer empty(0);
    require(!empty.data() && empty.capacity() == 0, "empty buffer");
    require(!empty.view(0, 0).data && empty.view().count == 0, "empty views");
    require_error<std::out_of_range>([&] { empty.view(0, 1); }, "out-of-range view");
    require_error<std::overflow_error>([&] { empty.view(SIZE_MAX, 0); }, "view offset overflow");
    require_error<std::overflow_error>([] { Buffer oversized(SIZE_MAX); }, "allocation size overflow");
    require_error<gpu::Error>([] { Buffer bad_device(0, -1); }, "negative device ordinal");
    require_error<gpu::Error>([] { Buffer bad_device(0, INT16_MAX + 1); }, "truncated device ordinal");
}

void check_uploads() {
    gpu::Stream stream(0);
    Buffer buffer(5, stream.device());
    const auto slots = buffer.view();
    require(slots.count == 5 && slots.where.kind == memory::domain::device
            && slots.where.device_ordinal == stream.device(), "full view capacity or placement");
    require(reinterpret_cast<uintptr_t>(slots.data) % alignof(Tile) == 0, "tile alignment");
    require(buffer.view(1, 2).data == slots.data + 1, "ranged view offset");
    require(buffer.view(5, 0).count == 0, "empty view at end");

    Tile source[2]{}, result[2]{}, last{};
    for (unsigned i = 0; i < Tile::rows * Tile::columns; ++i) {
        source[0].data()[i] = __float2half(float(i));
        source[1].data()[i] = __float2half(float(i + 256));
    }
    buffer.upload_async(source, 2, stream, 1);
    buffer.upload_async(source + 1, 1, stream, 4);
    buffer.upload_async(source, 1, stream, 4);

    const auto &read = buffer;
    require(read.view(4, 1).data == slots.data + 4, "const ranged view offset");
    read.download_async(result, 2, stream, 1);
    read.download_async(&last, 1, stream, 4);
    stream.synchronize();
    require(!std::memcmp(result, source, sizeof(source)), "write to another slot changed existing values");
    require(!std::memcmp(&last, source, sizeof(last)), "slot refresh");
    require_error<std::out_of_range>([&] { buffer.upload_async(source, 2, stream, 4); },
                                    "upload exceeds capacity");

    Buffer moved(static_cast<Buffer &&>(buffer));
    require(moved.data() == slots.data && moved.capacity() == 5, "move changed storage");
    require(!buffer.data() && buffer.capacity() == 0, "moved-from buffer");
}

} // namespace

int main(int argc, char **argv) try {
    check_empty();
    if (argc == 2 && !std::strcmp(argv[1], "--host-only")) return 0;
    int devices = 0;
    const auto status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver
        || (status == cudaSuccess && !devices)) return 77;
    gpu::check(status, "cudaGetDeviceCount");
    check_uploads();
    return 0;
} catch (const std::exception &error) {
    std::fprintf(stderr, "TiledBuffer: %s\n", error.what());
    return 1;
}
