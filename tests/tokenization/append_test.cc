#include <Cellerator/compute/tokenization/sequence_buffer.hh>

#include <ATen/ATen.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
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

} // namespace

int main() {
    using namespace cellerator::tokenization;
    try {
        auto longs = at::TensorOptions().dtype(at::kLong);
        auto floats = at::TensorOptions().dtype(at::kFloat);
        std::unordered_map<std::string, at::Tensor> examples{
            {"time", at::zeros({1}, floats)},
            {"coord", at::zeros({1, 2}, floats)}};
        auto buffer = SequenceBuffer::create({4, 1, 0}, examples);
        TokenBatch first{
            at::tensor({1, 2, 3, 4}, longs), at::tensor({0, 3, 4}, longs),
            {{"time", at::tensor({1.f, 2.f, 3.f, 4.f}, floats)},
             {"coord", at::tensor({1.f, 1.f, 2.f, 2.f, 3.f, 3.f, 4.f, 4.f}, floats)
                            .reshape({4, 2})}}};
        auto rows = at::tensor({0, 1}, longs);
        auto first_result = buffer.append_into(rows, first);
        check(values<bool>(first_result.succeeded) == std::vector<bool>({true, true}),
              "first append should succeed for rows with sufficient capacity");
        check(values<std::int64_t>(buffer.lengths()) == std::vector<std::int64_t>({3, 1, 0}),
              "append should update only selected sequence lengths");

        TokenBatch overflow{
            at::tensor({5, 6, 7}, longs), at::tensor({0, 2, 3}, longs),
            {{"time", at::tensor({5.f, 6.f, 7.f}, floats)},
             {"coord", at::tensor({5.f, 5.f, 6.f, 6.f, 7.f, 7.f}, floats)
                            .reshape({3, 2})}}};
        auto result = buffer.append_into(at::tensor({0, 1}, longs), overflow);
        check(values<bool>(result.succeeded) == std::vector<bool>({false, false}),
              "overflow must be reported per sequence");
        check(values<std::int64_t>(buffer.lengths()) == std::vector<std::int64_t>({3, 1, 0}),
              "overflowed appends must not change lengths");
        auto compact = buffer.compact();
        check(values<std::int64_t>(compact.token_ids) == std::vector<std::int64_t>({1, 2, 3, 4}),
              "overflow must not partially modify history");
        check(values<float>(compact.attributes.at("time"))
                  == std::vector<float>({1.f, 2.f, 3.f, 4.f}),
              "overflow must preserve all attribute storage");

        auto bigger = buffer.reserve({5, 2, 2});
        TokenBatch last{
            at::tensor({5, 6}, longs), at::tensor({0, 1, 2}, longs),
            {{"time", at::tensor({5.f, 6.f}, floats)},
             {"coord", at::tensor({5.f, 5.f, 6.f, 6.f}, floats).reshape({2, 2})}}};
        auto grown = bigger.append_into(at::tensor({0, 2}, longs), last);
        check(values<bool>(grown.succeeded) == std::vector<bool>({true, true}),
              "explicit reserve should permit later appends");
        check(values<std::int64_t>(bigger.compact().token_ids)
                  == std::vector<std::int64_t>({1, 2, 3, 5, 4, 6}),
              "explicit reserve and append should preserve each row's history");

        auto validation_buffer = SequenceBuffer::create({4, 4});
        bool rejected = false;
        try {
            validation_buffer.append_checked(at::tensor({0, 1}, longs),
                TokenBatch{at::tensor({5, 6, 7, 8}, longs),
                           at::tensor({0, 3, 1}, longs), {}});
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected && values<std::int64_t>(validation_buffer.lengths())
                              == std::vector<std::int64_t>({0, 0}),
              "malformed offsets must fail before any row is modified");
        rejected = false;
        try {
            validation_buffer.append_checked(at::tensor({0, 1}, longs),
                TokenBatch{at::tensor({-1}, longs), at::tensor({0, 1, 1}, longs), {}});
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected && values<std::int64_t>(validation_buffer.lengths())
                              == std::vector<std::int64_t>({0, 0}),
              "negative token IDs must fail before any row is modified");

        auto bool_example = at::empty({0}, at::TensorOptions().dtype(at::kBool));
        auto alias_buffer = SequenceBuffer::create({2}, {{"flag", bool_example}});
        auto alias_status = at::full({1}, true, at::TensorOptions().dtype(at::kBool));
        rejected = false;
        try {
            alias_buffer.append_into(at::tensor({0}, longs),
                TokenBatch{at::tensor({9}, longs), at::tensor({0, 1}, longs),
                           {{"flag", alias_status}}}, alias_status);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected && values<bool>(alias_status) == std::vector<bool>({true})
                          && values<std::int64_t>(alias_buffer.lengths())
                                 == std::vector<std::int64_t>({0}),
              "status buffers must not alias append payloads or be overwritten on rejection");

        auto source_alias_buffer = SequenceBuffer::create({2});
        rejected = false;
        try {
            source_alias_buffer.append_into(at::tensor({0}, longs),
                TokenBatch{source_alias_buffer.token_storage(), at::tensor({0, 1}, longs), {}});
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected && values<std::int64_t>(source_alias_buffer.lengths())
                              == std::vector<std::int64_t>({0}),
              "input payloads must not alias destination storage");

        auto empty_attribute = at::empty({0, 0}, floats);
        auto empty_attribute_buffer = SequenceBuffer::create({2}, {{"empty", empty_attribute}});
        auto empty_append = empty_attribute_buffer.append_into(at::tensor({0}, longs),
            TokenBatch{at::tensor({4}, longs), at::tensor({0, 1}, longs),
                       {{"empty", at::empty({1, 0}, floats)}}});
        check(values<bool>(empty_append.succeeded) == std::vector<bool>({true})
                  && values<std::int64_t>(empty_attribute_buffer.lengths())
                         == std::vector<std::int64_t>({1}),
              "zero-width trailing attribute dimensions must not disrupt append");
        auto zero_token_buffer = SequenceBuffer::create({0}, {{"empty", empty_attribute}});
        auto zero_token_append = zero_token_buffer.append_into(at::tensor({0}, longs),
            TokenBatch{at::empty({0}, longs), at::tensor({0, 0}, longs),
                       {{"empty", at::empty({0, 0}, floats)}}});
        check(values<bool>(zero_token_append.succeeded) == std::vector<bool>({true})
                  && values<std::int64_t>(zero_token_buffer.lengths())
                         == std::vector<std::int64_t>({0}),
              "empty payloads with zero-width attributes must append without reshaping");

        auto corrupted = SequenceBuffer::create({2});
        corrupted.lengths().fill_(-1);
        rejected = false;
        try {
            corrupted.append_checked(at::tensor({0}, longs),
                TokenBatch{at::tensor({1}, longs), at::tensor({0, 1}, longs), {}});
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, "append must reject externally corrupted negative lengths");
        rejected = false;
        try {
            (void)corrupted.compact();
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, "compaction must reject externally corrupted lengths");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
