#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/nn.hh>

#include <cassert>

int main() {
    cellerator::nn::Parameters parameters(at::Device(at::kCPU), at::kFloat);
    assert(parameters.size() == 0);
}
