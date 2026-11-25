#include <Cellerator/memory/striped/StripedBuffer.hh>
#include <Cellerator/tensor/striped.hh>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace memory = cellerator::memory;
namespace gpu = memory::cuda;
using namespace cellerator::tensor::striped;
using Layout = StripLayout<float, uint16_t, StripOrder::slot_major>;
using View = StripView<float, uint16_t, StripOrder::slot_major>;
using ConstView = StripView<const float, const uint16_t, StripOrder::slot_major>;

static void require(bool condition, const char *label) {
    if (!condition) {
        std::fprintf(stderr, "StripedBuffer CUDA: %s\n", label);
        std::exit(1);
    }
}

__global__ void read_stripes(ConstView a, ConstView b, float *out) {
    unsigned row = threadIdx.x;
    unsigned r = a.packed_row(row);
    if (r != View::missing_row) {
        size_t i = a.slot(r, 1), j = b.slot(b.packed_row(row), 2);
        out[r] = a.values[i] + b.values[j] + a.column_indices[i] + b.column_indices[j];
    }
}

int main() try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    Layout layout;
    require(Layout::prepare({(1u << 3) | (1u << 27), 0, 4, 4}, &layout)
        == memory::status::success, "prepare");
    gpu::PinnedBuffer host(4 * layout.bytes());
    auto *bytes = static_cast<unsigned char *>(host.data());
    std::memset(bytes, 0, host.bytes());
    View host_strips[4];
    for (unsigned i = 0; i < 4; ++i)
        require(layout.bind(bytes + i * layout.bytes(), layout.bytes(), &host_strips[i]) ==
            memory::status::success, "host stripe");
    for (unsigned r = 0; r < 2; ++r) {
        size_t i = host_strips[0].slot(r, 1), j = host_strips[1].slot(r, 2);
        host_strips[0].values[i] = float(r + 1);
        host_strips[0].column_indices[i] = 64000;
        host_strips[1].values[j] = 2;
        host_strips[1].column_indices[j] = 3;
        host_strips[2].values[i] = float(r + 9);
        host_strips[2].column_indices[i] = 60000;
        host_strips[3].values[j] = 4;
        host_strips[3].column_indices[j] = 5;
    }

    float result[4]{};
    memory::StripedBuffer buffer(3 * layout.bytes());
    gpu::DeviceBuffer output(sizeof(result), buffer.device());
    gpu::Stream copy(buffer.device()), compute(buffer.device());
    gpu::Event first_ready(buffer.device()), first_read(buffer.device()), next_ready(buffer.device());
    const size_t span = layout.bytes(), shared_offset = 2 * span;
    const auto &input = buffer;
    ConstView a, b0, b1;
    require(input.view(shared_offset, layout, &a) == memory::status::success &&
        input.view(0, layout, &b0) == memory::status::success &&
        input.view(span, layout, &b1) == memory::status::success, "disjoint device stripes");
    buffer.upload_async(bytes, span, copy, shared_offset);
    buffer.upload_async(bytes + span, span, copy);
    first_ready.record(copy);

    auto *device_output = static_cast<float *>(output.data());
    compute.wait(first_ready);
    read_stripes<<<1, 32, 0, compute.get()>>>(a, b0, device_output);
    gpu::check(cudaGetLastError(), "read first stripes");
    first_read.record(compute);

    buffer.upload_async(bytes + 3 * span, span, copy, span);
    copy.wait(first_read);
    buffer.upload_async(bytes + 2 * span, span, copy, shared_offset);
    next_ready.record(copy);
    compute.wait(next_ready);
    read_stripes<<<1, 32, 0, compute.get()>>>(a, b1, device_output + 2);
    gpu::check(cudaGetLastError(), "read next stripes");
    output.download_async(result, sizeof(result), compute);
    compute.synchronize();
    require(result[0] == 64006 && result[1] == 64007, "device view addressing and narrow loads");
    require(result[2] == 60018 && result[3] == 60019, "prefetch and reuse with stream events");
    std::puts("StripedBuffer device addressing and stream-ordered reuse passed");
} catch (const std::exception &error) {
    std::fprintf(stderr, "StripedBuffer CUDA: %s\n", error.what());
    return 1;
}
