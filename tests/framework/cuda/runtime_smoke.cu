#include <Cellerator/modeling/cuda/plans.hh>
#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>
using namespace cellerator::modeling::cuda;

int main(int argc, char** argv) {
    assert(checked_bytes(7, 4) == 28);
    bool rejected = false;
    try { (void)checked_bytes(std::numeric_limits<std::size_t>::max(), 2); }
    catch (const std::overflow_error&) { rejected = true; }
    assert(rejected);
    DeviceBuffer empty(0, 0);
    assert(empty.bytes() == 0 && empty.view().data == nullptr);
    auto borrowed = Stream::borrow(nullptr, 0);
    assert(!borrowed.owns_handle());
    const DeviceView fake{reinterpret_cast<void*>(std::uintptr_t{4096}), 64, 0};
    const auto tail = fake.subview(64, 0);
    assert(tail.data == reinterpret_cast<void*>(std::uintptr_t{4160}));
    assert(tail.bytes == 0 && tail.device == 0);
    rejected = false;
    try { (void)fake.subview(65, 0); }
    catch (const std::out_of_range&) { rejected = true; }
    assert(rejected);
    rejected = false;
    try { require_view(ConstDeviceView{nullptr, 8, 0}, 8, 0, "null test"); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    rejected = false;
    try { require_view(ConstDeviceView{fake.data, fake.bytes, 1}, 8, 0, "device test"); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    assert(item_size(DType::float16) == sizeof(std::uint16_t));
    assert(item_size(DType::float32) == sizeof(float));
    rejected = false;
    try { DispatchPlan bad(1, {0}, {1.f}, {0, 1}, {1}, {0, 1}, borrowed); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    if (argc == 2 && std::strcmp(argv[1], "--host-only") == 0) {
        std::cout << "runtime host contract checks passed (no CUDA calls)\n";
        return 0;
    }
    Stream stream(0), consumer(0);
    PinnedBuffer source(6 * sizeof(float)), output(6 * sizeof(float));
    const std::vector<float> x{1, 2, 3, 4, 5, 6};
    std::memcpy(source.data(), x.data(), source.bytes());
    DeviceBuffer input(source.bytes() + 8, 0);
    input.upload_async(source.data(), source.bytes(), stream, 4);
    input.upload_async(nullptr, 0, stream, input.bytes());
    rejected = false;
    try { input.upload_async(source.data(), 1, stream, input.bytes()); }
    catch (const std::out_of_range&) { rejected = true; }
    assert(rejected);
    DeviceBuffer copied(source.bytes(), 0);
    copied.copy_from_async(input.view().subview(4, source.bytes()), stream);
    Event produced(0);
    produced.record(stream);
    consumer.wait(produced);
    Event begin(0, true), end(0, true);
    begin.record(consumer);
    copied.download_async(output.data(), output.bytes(), consumer);
    end.record(consumer);
    consumer.synchronize();
    assert(produced.ready());
    assert(begin.elapsed_ms(end) >= 0.0f);
    std::vector<float> copied_values(6);
    std::memcpy(copied_values.data(), output.data(), output.bytes());
    assert(copied_values == x);
    // Dispatch order repeats origin 0, uses origin 2, and leaves origin 1 idle.
    DispatchPlan plan(3, {0, 2, 0}, {0.25f, 2.f, 0.75f}, {0, 2, 2, 3}, {0, 2, 1}, {0, 2, 3}, stream);
    DeviceBuffer dispatched(6 * sizeof(float), 0), result(6 * sizeof(float), 0);
    const auto input_view = input.view().subview(4, source.bytes());
    plan.gather(input_view, dispatched.view(), 2, DType::float32, stream);
    plan.combine(dispatched.view(), result.view(), 2, DType::float32, stream);
    result.download_async(output.data(), result.bytes(), stream); stream.synchronize();
    std::vector<float> actual(6);
    std::memcpy(actual.data(), output.data(), output.bytes());
    const std::vector<float> expected{1, 2, 0, 0, 10, 12};
    assert(actual == expected);
    auto profile = std::make_shared<QuantizationProfile>(std::vector<std::uint32_t>{0, 2, 4}, std::vector<float>{0, 4, 0, 8}, std::vector<std::uint32_t>{0, 1}, 1, stream);
    QuantizedState packed(profile, 3);
    packed.encode(input_view, DType::float32, {}, stream);
    packed.decode(result.view(), DType::float32, stream);
    assert(packed.status(stream) == 0);
    result.download_async(output.data(), result.bytes(), stream); stream.synchronize();
    std::memcpy(actual.data(), output.data(), output.bytes());
    assert((actual == std::vector<float>{0, 0, 4, 0, 4, 8}));
    Event done(0); done.record(stream); done.synchronize(); assert(done.ready());
    std::cout << "runtime CUDA smoke passed\n";
}
