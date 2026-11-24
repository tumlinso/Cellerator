#include <Cellerator/compute/tokenization/tokenization.hh>

#include <ATen/ATen.h>

#include <stdexcept>

int main() {
    using namespace cellerator::tokenization;
    auto vocabulary = make_vocabulary(
        at::tensor({11, 12}, at::TensorOptions().dtype(at::kLong)),
        at::tensor({2, 5}, at::TensorOptions().dtype(at::kLong)));
    auto events = event_stream(
        at::tensor({12, 11, 12}, at::TensorOptions().dtype(at::kLong)),
        at::tensor({0, 3}, at::TensorOptions().dtype(at::kLong)),
        vocabulary);
    if (events.token_ids.numel() != 3 || events.offsets.numel() != 2
        || events.token_ids[0].item<std::int64_t>() != 5
        || events.token_ids[1].item<std::int64_t>() != 2
        || events.token_ids[2].item<std::int64_t>() != 5)
        throw std::runtime_error("installed event-stream consumer failed");
    return 0;
}
