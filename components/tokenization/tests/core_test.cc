#include <Cellerator/compute/tokenization/tokenization.hh>

#include <ATen/ATen.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename T>
std::vector<T> values(const at::Tensor& tensor) {
    auto host = tensor.to(at::kCPU).contiguous();
    std::vector<T> out;
    out.reserve(host.numel());
    for (std::int64_t i = 0; i < host.numel(); ++i) out.push_back(host.data_ptr<T>()[i]);
    return out;
}

void event_and_boundaries() {
    using namespace cellerator::tokenization;
    auto options = at::TensorOptions().dtype(at::kLong);
    auto vocabulary = make_vocabulary(
        at::tensor({10, 20, 30, 40}, options),
        at::tensor({1, 2, 3, 4}, options));
    auto ids = at::tensor({20, 10, 20, 30}, options);
    auto offsets = at::tensor({0, 3, 4}, options);
    auto time = at::tensor({0.2f, 0.1f, 0.8f, 1.4f},
                           at::TensorOptions().dtype(at::kFloat));
    auto events = event_stream(ids, offsets, vocabulary, {{"time", time}});
    check(values<std::int64_t>(events.token_ids) == std::vector<std::int64_t>({2, 1, 2, 3}),
          "event stream must preserve supplied order and repetitions");

    TokenBatch prefix{
        at::tensor({90, 90}, options), at::tensor({0, 1, 2}, options),
        {{"time", at::zeros({2}, time.options())}}};
    TokenBatch suffix{
        at::tensor({91, 91}, options), at::tensor({0, 1, 2}, options),
        {{"time", at::ones({2}, time.options())}}};
    auto bounded = add_boundaries(events, prefix, suffix);
    check(values<std::int64_t>(bounded.token_ids)
              == std::vector<std::int64_t>({90, 2, 1, 2, 91, 90, 3, 91}),
          "caller-supplied prefix and suffix tokens must bracket each sequence");
    check(values<std::int64_t>(bounded.offsets)
              == std::vector<std::int64_t>({0, 5, 8}),
          "boundary composition must preserve ragged offsets");
    auto padded = pad_sequences(bounded, 0);
    check(padded.token_ids.sizes() == at::IntArrayRef({2, 5}),
          "ragged sequences should pad to the longest row");
    auto packed = pack_padded(padded.token_ids, padded.lengths, padded.attributes);
    check(at::equal(packed.token_ids, bounded.token_ids)
              && at::equal(packed.offsets, bounded.offsets)
              && at::equal(packed.attributes.at("time"), bounded.attributes.at("time")),
          "pad/pack must preserve token and attribute ordering");
}

void vocabulary_drop_and_gradient() {
    using namespace cellerator::tokenization;
    auto longs = at::TensorOptions().dtype(at::kLong);
    auto vocab = make_vocabulary(at::tensor({1, 2}, longs), at::tensor({7, 8}, longs),
                                 unknown_identity_policy::drop);
    auto input = at::tensor({1, 9, 2, 9}, longs);
    auto offsets = at::tensor({0, 2, 4}, longs);
    auto attr = at::tensor({1.f, 10.f, 2.f, 20.f},
        at::TensorOptions().dtype(at::kFloat)).set_requires_grad(true);
    auto output = event_stream(input, offsets, vocab, {{"x", attr}});
    check(values<std::int64_t>(output.token_ids) == std::vector<std::int64_t>({7, 8}),
          "drop policy must filter unknown identities");
    check(values<std::int64_t>(output.offsets) == std::vector<std::int64_t>({0, 1, 2}),
          "drop policy must update every ragged boundary");
    check(output.attributes.at("x").requires_grad(),
          "gathered continuous attributes should retain ATen autograd");
    output.attributes.at("x").sum().backward();
    check(values<float>(attr.grad()) == std::vector<float>({1.f, 0.f, 1.f, 0.f}),
          "attribute gradient should follow retained source occurrences");
}

void ranked_features() {
    using namespace cellerator::tokenization;
    auto longs = at::TensorOptions().dtype(at::kLong);
    auto vocab = make_vocabulary(at::tensor({10, 20, 30, 40}, longs),
                                 at::tensor({10, 20, 30, 40}, longs));
    auto input_values = at::tensor({1.f, 2.f, 0.f, 4.f,
                                    2.f, 3.f, 4.f, 0.f},
                                   at::TensorOptions().dtype(at::kFloat)).reshape({2, 4});
    auto scores = at::tensor({1.f, 5.f, 0.f, 5.f,
                              2.f, 3.f, 4.f, 0.f},
                             at::TensorOptions().dtype(at::kFloat)).reshape({2, 4});
    RankedFeatureOptions config;
    config.max_per_row = 2;
    config.scores = scores;
    config.retain_measurements = true;
    config.retain_scores = true;
    auto output = ranked_feature(input_values, at::tensor({40, 20, 30, 10}, longs),
                                 vocab, config);
    check(values<std::int64_t>(output.token_ids)
              == std::vector<std::int64_t>({10, 20, 30, 20}),
          "ranked features must use deterministic feature-ID tie order");
    check(values<std::int64_t>(output.offsets)
              == std::vector<std::int64_t>({0, 2, 4}),
          "ranked features must retain requested number per row");
    check(values<float>(output.attributes.at("measurement"))
              == std::vector<float>({4.f, 2.f, 4.f, 3.f}),
          "selected measurements must remain aligned with tokens");
    check(values<float>(output.attributes.at("score"))
              == std::vector<float>({5.f, 5.f, 4.f, 3.f}),
          "selected scores must remain aligned with tokens");
}

void drop_unknown_before_truncation() {
    using namespace cellerator::tokenization;
    auto longs = at::TensorOptions().dtype(at::kLong);
    auto vocabulary = make_vocabulary(at::tensor({20}, longs), at::tensor({9}, longs),
                                      unknown_identity_policy::drop);
    auto input_values = at::tensor({100.f, 8.f}, at::TensorOptions().dtype(at::kFloat)).reshape({1, 2});
    RankedFeatureOptions options;
    options.max_per_row = 1;
    auto output = ranked_feature(input_values, at::tensor({10, 20}, longs), vocabulary, options);
    check(values<std::int64_t>(output.token_ids) == std::vector<std::int64_t>({9}),
          "unknown features must be dropped before top-k truncation");
}

void geneformer_style() {
    using namespace cellerator::tokenization;
    auto longs = at::TensorOptions().dtype(at::kLong);
    auto doubles = at::TensorOptions().dtype(at::kDouble);
    auto ids = at::tensor({100, 200, 300}, longs);
    auto vocab = make_vocabulary(at::tensor({100, 200}, longs), at::tensor({1, 2}, longs),
                                 unknown_identity_policy::drop);
    auto counts = at::tensor({20., 10., 0., 100., 50., 30.}, doubles).reshape({2, 3});
    auto total = at::tensor({100., 100.}, doubles);
    auto medians = at::tensor({10., 20., std::numeric_limits<double>::quiet_NaN()}, doubles);
    auto result = geneformer_style(counts, total, medians, ids, vocab, 2);
    check(values<std::int64_t>(result.token_ids) == std::vector<std::int64_t>({1, 2, 1, 2}),
          "Geneformer profile should rank positive counts by corpus median-scaled score");

    auto crow = at::tensor({0, 1}, longs);
    auto columns = at::tensor({0}, longs);
    auto explicit_zero = at::tensor({0.}, doubles);
    auto sparse_zeros = at::sparse_csr_tensor(crow, columns, explicit_zero,
        at::IntArrayRef({1, 3}), at::TensorOptions().dtype(at::kDouble));
    auto empty = geneformer_style(sparse_zeros, at::tensor({0.}, doubles), medians,
                                  ids, vocab, 2);
    check(empty.token_ids.numel() == 0
              && values<std::int64_t>(empty.offsets) == std::vector<std::int64_t>({0, 0}),
          "explicit-zero CSR entries must not make an observation nonempty");
}

void geneformer_compute_precision_policy() {
    using namespace cellerator::tokenization;
    auto longs = at::TensorOptions().dtype(at::kLong);
    auto floats = at::TensorOptions().dtype(at::kFloat);
    auto doubles = at::TensorOptions().dtype(at::kDouble);
    auto feature_ids = at::tensor({20, 10}, longs);
    auto vocab = make_vocabulary(at::tensor({10, 20}, longs),
                                 at::tensor({1, 2}, longs));
    auto counts = at::tensor({1.f, 1.f}, floats).reshape({1, 2});
    auto totals = at::tensor({1.f}, floats);
    auto medians = at::tensor({1.0, 1.00000001}, doubles);
    auto default_precision = geneformer_style(
        counts, totals, medians, feature_ids, vocab, 2);
    auto explicit_fp64 = geneformer_style(
        counts, totals, medians, feature_ids, vocab, 2, {}, at::kDouble);
    check(values<std::int64_t>(default_precision.token_ids)
              == std::vector<std::int64_t>({1, 2}),
          "Geneformer profile should use FP32 arithmetic by default for FP32 counts");
    check(values<std::int64_t>(explicit_fp64.token_ids)
              == std::vector<std::int64_t>({2, 1}),
          "Geneformer profile should preserve precision when FP64 is explicitly requested");
}

} // namespace

int main() {
    try {
        event_and_boundaries();
        vocabulary_drop_and_gradient();
        ranked_features();
        drop_unknown_before_truncation();
        geneformer_style();
        geneformer_compute_precision_policy();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
