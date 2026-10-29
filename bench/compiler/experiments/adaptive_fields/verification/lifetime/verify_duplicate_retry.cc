// Exercise the real experiment-local worker after its completion is consumed.
#define main lifetime_probe_main
#include "../../cpu/runtime/lifetime_probe.cc"
#undef main

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::atomic<unsigned> closed{0};
    Worker worker(argv[1], &closed);
    Request request{0, 1, 1, 2.0f, false, "same-structure-contract"};
    const auto first = worker.submit(request);
    if (!first.accepted || first.duplicate) return 3;
    const auto completion = worker.wait(first.id);
    if (completion.failed || !completion.module) return 4;
    const auto retry = worker.submit(request);
    std::cout << "first_id=" << first.id << " completion_module="
              << static_cast<bool>(completion.module) << " retry_id=" << retry.id
              << " retry_duplicate=" << retry.duplicate
              << " retry_accepted=" << retry.accepted << '\n';
    return !(retry.accepted && retry.duplicate && retry.id == first.id);
}
