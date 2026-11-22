#include "update_fixture.cuh"
int main(int argc, char** argv) try {
    int count = 0;
    cuda_ok(cudaGetDeviceCount(&count));
    require(count == 1, "one leased GPU required");
    if (argc == 3 && !std::strcmp(argv[1], "--fp32-trace")) {
        update_test(true, false);
        fp32_trace_update_test(argv[2]);
        return 0;
    }
    if (argc == 4 && !std::strcmp(argv[1], "--fp32-bench")) {
        const bool mapped = !std::strcmp(argv[2], "cusparse-mapped");
        const bool vendor = mapped || !std::strcmp(argv[2], "cusparse");
        const bool prepared_vendor = !std::strcmp(argv[2], "prepared-cusparse");
        require(prepared_vendor || vendor || !std::strcmp(argv[2], "native"), "unknown FP32 trace backend");
        fp32_trace_update_test(argv[3], true, vendor, mapped, prepared_vendor);
        return 0;
    }
    require(argc == 1, "usage: ce_nf1_v06 [--fp32-trace path | --fp32-bench backend path]");
    update_test(false);
    update_test(true, true);
    update_test(true, false);
    std::cout << "V06 real f16/f32 delta and gradient updates, response invalidation and independent branching passed\n";
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
