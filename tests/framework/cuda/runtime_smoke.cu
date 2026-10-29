#include <Cellerator/modeling/cuda/plans.hh>
#include <cassert>
#include <cmath>
#include <cstring>
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
    rejected = false;
    try { DispatchPlan bad(1, {0}, {1.f}, {0, 1}, {1}, {0, 1}, borrowed); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    if (argc == 2 && std::strcmp(argv[1], "--host-only") == 0) {
        std::cout << "runtime host contract checks passed (no CUDA calls)\n";
        return 0;
    }
    Stream stream(0);
    std::vector<float> x{1, 2, 3, 4, 5, 6};
    DeviceBuffer input(x.size() * sizeof(float), 0);
    input.upload_async(x.data(), input.bytes(), stream);
    // Dispatch order repeats origin 0, uses origin 2, and leaves origin 1 idle.
    DispatchPlan plan(3, {0, 2, 0}, {0.25f, 2.f, 0.75f}, {0, 2, 2, 3}, {0, 2, 1}, {0, 2, 3}, stream);
    DeviceBuffer dispatched(6 * sizeof(float), 0), result(6 * sizeof(float), 0);
    plan.gather(input.view(), dispatched.view(), 2, DType::float32, stream);
    plan.combine(dispatched.view(), result.view(), 2, DType::float32, stream);
    std::vector<float> actual(6);
    result.download_async(actual.data(), result.bytes(), stream); stream.synchronize();
    const std::vector<float> expected{1, 2, 0, 0, 10, 12};
    assert(actual == expected);
    auto profile = std::make_shared<QuantizationProfile>(std::vector<std::uint32_t>{0, 2, 4}, std::vector<float>{0, 4, 0, 8}, std::vector<std::uint32_t>{0, 1}, 1, stream);
    QuantizedState packed(profile, 3);
    packed.encode(input.view(), DType::float32, {}, stream);
    packed.decode(result.view(), DType::float32, stream);
    assert(packed.status(stream) == 0);
    result.download_async(actual.data(), result.bytes(), stream); stream.synchronize();
    assert((actual == std::vector<float>{0, 0, 4, 0, 4, 8}));
    Event done(0); done.record(stream); done.synchronize(); assert(done.ready());
    std::cout << "runtime CUDA smoke passed\n";
}
