#include "workload.hh"

int main(int argc, char **argv) {
    using namespace sequence_bench;
    const Settings s = settings(argc, argv);
    const Input input = make_input(s.bases);
    const double pack_start = now_ms();
    const auto view = pack(input, s.bases);
    const double pack_ms = now_ms() - pack_start;
    const double prepare_start = now_ms();
    const auto motif = prepare_predicate();
    const double prepare_ms = now_ms() - prepare_start;

    uint64_t count = 0u;
    for (uint32_t i = 0; i < s.warmup; ++i)
        require(seq::count_motif(view, motif, &count) == seq::Status::ok, "warmup motif count");
    const double start = now_ms();
    for (uint32_t i = 0; i < s.repeats; ++i)
        require(seq::count_motif(view, motif, &count) == seq::Status::ok, "repeated motif count");
    const double repeat_ms = now_ms() - start;
    require(count == input.expected, "host count differs from text oracle");
    print_result(s, "cpu", count, pack_ms, prepare_ms, 0.0, repeat_ms);
    free(input.words); free(input.bases);
    return 0;
}
