#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/nn/query.hh>

#include <cassert>
#include <tuple>
#include <vector>

namespace nn = cellerator::nn;

struct Input { int value; };

struct BaseModel {
    int* calls;
    int operator()(const Input& input, nn::Run run) const {
        ++*calls;
        return input.value + (run.training ? 100 : 0);
    }
};

struct Coordinates { int row; };

struct Decoder {
    int query(int source, const nn::PreparedTargets<int>& targets,
              const Coordinates& coordinates, nn::Run run) const {
        return source + targets.output_columns().front() + coordinates.row +
            (run.training ? 1000 : 0);
    }
};

int main() {
    int calls = 0;
    const BaseModel model{&calls};
    Input input{5};

    const auto binding = cellerator::biology::FeatureBinding<int>::from_ids({10, 20}, {0, 1});
    const at::Device device(at::kCPU);
    const auto first_targets = nn::prepare_targets(binding, std::vector<int>{20}, device);
    const auto second_targets = nn::prepare_targets(binding, std::vector<int>{10}, device);
    Decoder decoder;
    auto selector = [](int result, const Input& input) {
        return result * 2 + input.value;
    };
    auto readout = nn::readout(decoder, selector, binding);
    Coordinates first_coordinates{4};
    Coordinates second_coordinates{7};
    auto first_request = nn::query(readout, first_targets, first_coordinates);
    auto second_request = nn::query(readout, second_targets, second_coordinates);

    auto both = nn::predict(model, input, first_request, second_request);
    static_assert(std::tuple_size_v<decltype(both)> == 2);
    assert(calls == 1);
    const auto& [first, second] = both;
    assert(first.values == 20 && second.values == 22);
    assert(first.payload.feature_ids.size() == 1 && first.payload.feature_ids[0] == 20);
    assert(second.payload.feature_ids.size() == 1 && second.payload.feature_ids[0] == 10);
    assert(first.payload.coordinates.get().row == 4);
    assert(second.payload.coordinates.get().row == 7);

    input.value = 8;
    auto selected = nn::predict(model, input, first_request);
    assert(calls == 2 && selected.values == 29);

    auto training = nn::predict(model, input, nn::Run{true}, first_request);
    assert(calls == 3 && training.values == 1229);
}
