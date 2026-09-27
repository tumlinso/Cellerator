#include <CelleraTorch/mechanism.hh>

#include <torch/optim/adam.h>
#include <torch/serialize.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace {
struct DenseMechanismNet final : torch::nn::Module {
    DenseMechanismNet(c10::intrusive_ptr<celleratorch::Mechanism> handle)
        : pre(register_module("pre", torch::nn::Linear(2, 2))),
          left(register_module("left", std::make_shared<celleratorch::MechanismModule>(handle))),
          right(register_module("right", std::make_shared<celleratorch::MechanismModule>(handle))),
          post(register_module("post", torch::nn::Linear(1, 1))),
          mechanism(std::move(handle)) {}

    torch::Tensor forward(const torch::Tensor& input, const std::vector<std::int64_t>& axis) {
        auto a = pre->forward(input);
        auto b = pre->forward(input);
        return post->forward(left->forward(a, axis) + right->forward(b, axis));
    }
    torch::nn::Linear pre{nullptr}, post{nullptr};
    std::shared_ptr<celleratorch::MechanismModule> left, right;
    c10::intrusive_ptr<celleratorch::Mechanism> mechanism;
};

std::vector<torch::Tensor> trainable_params(DenseMechanismNet& net) {
    auto values = net.pre->parameters();
    auto post = net.post->parameters();
    values.insert(values.end(), post.begin(), post.end());
    values.push_back(net.mechanism->coefficients());
    return values;
}
}

int main() {
    if (!torch::cuda::is_available()) {
        std::cerr << "FAIL: CUDA is required for the CelleraTorch mechanism acceptance test\n";
        return 1;
    }
    const auto device = torch::Device(torch::kCUDA, 0);
    auto options = torch::TensorOptions().device(device).dtype(torch::kFloat32);
    // Three axes encoded as domain/order/geometry/partition low/high pairs.
    std::vector<std::int64_t> axes{
        11, 1, 12, 1, 13, 1, 14, 1,
        21, 1, 22, 1, 23, 1, 24, 1,
        31, 1, 32, 1, 33, 1, 34, 1};
    auto make_mechanism = [&](torch::Tensor initial, std::int64_t precision) {
      return c10::make_intrusive<celleratorch::Mechanism>(
        std::move(initial), axes, std::vector<std::int64_t>{2, 1, 1},
        std::vector<std::int64_t>{201, 1}, std::vector<std::int64_t>{301, 1},
        std::vector<std::int64_t>{0, 2}, std::vector<std::int64_t>{0, 1},
        std::vector<std::int64_t>{401, 1, 402, 1}, std::vector<std::int64_t>{0, 0},
        std::vector<std::int64_t>{0, 1}, std::vector<std::int64_t>{0},
        std::vector<std::int64_t>{0, 1}, std::vector<std::int64_t>{0},
        std::vector<std::int64_t>{403, 1}, std::vector<std::int64_t>{0},
        std::vector<std::int64_t>{0}, std::vector<std::int64_t>{501, 1},
        std::vector<double>{1.0}, 8, 4, precision);
    };
    bool rejected_half_overflow = false;
    try { (void)make_mechanism(torch::tensor({100000.0f}, options), 1); }
    catch (const c10::Error&) { rejected_half_overflow = true; }
    assert(rejected_half_overflow);

    auto mixed = make_mechanism(torch::tensor({0.5f}, options), 1);
    auto mixed_checkpoint = mixed->snapshot();
    const auto mixed_generation = mixed->generation();
    bool rejected_bad_restore = false;
    try { mixed->restore(torch::tensor({100000.0f}, options)); }
    catch (const c10::Error&) { rejected_bad_restore = true; }
    assert(rejected_bad_restore);
    assert(mixed->generation() == mixed_generation);
    assert(torch::allclose(mixed->snapshot(), mixed_checkpoint));
    auto mixed_k = mixed->coefficients();
    torch::optim::Adam explosive_adam({mixed_k}, torch::optim::AdamOptions(1.0e6));
    { torch::NoGradGuard no_grad; mixed_k.mutable_grad() = torch::ones_like(mixed_k); }
    bool rejected_half_overflow_step = false;
    try { celleratorch::guarded_adam_step(explosive_adam, {mixed}); }
    catch (const c10::Error&) { rejected_half_overflow_step = true; }
    assert(rejected_half_overflow_step && mixed->poisoned());
    assert(mixed->generation() == mixed_generation);
    mixed->restore(mixed_checkpoint); // explicit native checkpoint recovery.
    auto large_f32 = make_mechanism(torch::tensor({100000.0f}, options), 0);
    assert(std::abs(large_f32->snapshot().item<float>() - 100000.0f) < 1.0f);

    auto mechanism = make_mechanism(torch::tensor({0.5f}, options), 0);

    celleratorch::MechanismModule first(mechanism);
    celleratorch::MechanismModule second(mechanism);
    auto x1 = torch::tensor({{2.0f, 3.0f}, {1.0f, 2.0f}}, options).set_requires_grad(true);
    auto x2 = torch::tensor({{1.0f, 4.0f}}, options).set_requires_grad(true);
    auto y1 = first.forward(x1, std::vector<std::int64_t>(axes.begin(), axes.begin() + 8));
    auto y2 = second.forward(x2, std::vector<std::int64_t>(axes.begin(), axes.begin() + 8));
    auto loss = y1.sum() + y2.sum();
    loss.backward();
    auto k = mechanism->coefficients();
    assert(k.grad().defined());
    assert(std::abs(k.grad().item<float>() - 12.0f) < 1e-4f);
    assert(torch::allclose(x1.grad(), torch::tensor({{1.5f, 1.0f}, {1.0f, 0.5f}}, options), 1e-5, 1e-5));

    auto checkpoint = mechanism->snapshot();
    auto ordinary = torch::tensor({1.0f}, options).set_requires_grad(true);
    torch::optim::Adam optimizer({k, ordinary}, torch::optim::AdamOptions(0.01));
    optimizer.zero_grad();
    { torch::NoGradGuard no_grad; ordinary.mutable_grad() = torch::ones_like(ordinary); }
    celleratorch::guarded_adam_step(optimizer, {mechanism});
    assert(mechanism->generation() == 0); // no native parameter value changed.
    assert(ordinary.item<float>() < 1.0f); // stock Adam still ran for ordinary parameters.
    const auto ordinary_after_step = ordinary.item<float>();
    {
        torch::NoGradGuard no_grad;
        k.mutable_grad() = torch::full_like(k, std::numeric_limits<float>::infinity());
        ordinary.mutable_grad() = torch::ones_like(ordinary);
    }
    celleratorch::guarded_adam_step(optimizer, {mechanism}, true);
    assert(mechanism->generation() == 0); // scaler skip has no publication.
    assert(ordinary.item<float>() == ordinary_after_step);
    optimizer.zero_grad();
    { torch::NoGradGuard no_grad; k.mutable_grad() = torch::full_like(k, 12.0f); }
    celleratorch::guarded_adam_step(optimizer, {mechanism});
    assert(mechanism->generation() == 1);
    assert(!torch::allclose(k.detach(), checkpoint.to(device)));

    mechanism->restore(checkpoint);
    assert(torch::allclose(k.detach(), checkpoint.to(device)));
    auto x3 = torch::tensor({{2.0f, 2.0f}}, options).set_requires_grad(true);
    first.forward(x3, std::vector<std::int64_t>(axes.begin(), axes.begin() + 8)).sum().backward();
    celleratorch::guarded_adam_step(optimizer, {mechanism});
    assert(mechanism->generation() == 3); // restore publishes a new generation.

    // Exercise an installed C++ consumer: dense layers surround the biology
    // mechanism, two module views share one CE leaf, and Adam trains all parts.
    auto net = std::make_shared<DenseMechanismNet>(mechanism);
    net->pre->to(device);
    net->post->to(device);
    {
        torch::NoGradGuard no_grad;
        net->pre->weight.copy_(torch::eye(2, options));
        net->pre->bias.zero_();
        net->post->weight.fill_(1.0);
        net->post->bias.zero_();
    }
    auto net_params = trainable_params(*net);
    torch::optim::Adam net_optimizer(net_params, torch::optim::AdamOptions(0.01));
    auto samples = torch::tensor({{0.2f, 0.4f}, {0.5f, 0.7f}, {0.8f, 0.3f}, {0.9f, 0.6f}}, options);
    auto target = 1.6f * samples.select(1, 0).unsqueeze(1) * samples.select(1, 1).unsqueeze(1);
    float before;
    {
        torch::NoGradGuard no_grad;
        before = torch::mse_loss(net->forward(samples,
            std::vector<std::int64_t>(axes.begin(), axes.begin() + 8)), target).item<float>();
    }
    TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess, "CUDA sync failed before training timing");
    const auto lifecycle_start = std::chrono::steady_clock::now();
    double optimizer_step_ms = 0.0;
    for (int step = 0; step < 40; ++step) {
        net_optimizer.zero_grad();
        auto prediction = net->forward(samples, std::vector<std::int64_t>(axes.begin(), axes.begin() + 8));
        torch::mse_loss(prediction, target).backward();
        TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess, "CUDA sync failed before guarded Adam timing");
        const auto step_start = std::chrono::steady_clock::now();
        celleratorch::guarded_adam_step(net_optimizer, {mechanism});
        TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess, "CUDA sync failed after guarded Adam timing");
        optimizer_step_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - step_start).count();
    }
    TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess, "CUDA sync failed after training timing");
    const auto lifecycle_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - lifecycle_start).count();
    torch::Tensor trained;
    float after;
    {
        torch::NoGradGuard no_grad;
        trained = net->forward(samples, std::vector<std::int64_t>(axes.begin(), axes.begin() + 8));
        after = torch::mse_loss(trained, target).item<float>();
    }
    assert(after < before * 0.6f);
    std::cout << "training_lifecycle_ms=" << lifecycle_ms
              << " guarded_adam_total_ms=" << optimizer_step_ms
              << " steps=40 initial_loss=" << before << " final_loss=" << after
              << " native_reserved_bytes=" << mechanism->program()->reserved_bytes()
              << " native_generation=" << mechanism->generation() << "\n";

    torch::serialize::OutputArchive model_archive, optimizer_archive;
    net->save(model_archive);
    net_optimizer.save(optimizer_archive);
    auto saved_coefficients = mechanism->snapshot();
    model_archive.save_to("celleratorch-mechanism-model.pt");
    optimizer_archive.save_to("celleratorch-mechanism-optimizer.pt");
    auto reloaded_handle = make_mechanism(saved_coefficients.to(device), 0);
    auto reloaded = std::make_shared<DenseMechanismNet>(reloaded_handle);
    reloaded->pre->to(device);
    reloaded->post->to(device);
    torch::serialize::InputArchive model_input, optimizer_input;
    model_input.load_from("celleratorch-mechanism-model.pt");
    optimizer_input.load_from("celleratorch-mechanism-optimizer.pt");
    reloaded->load(model_input);
    // Module loading copies the serialized leaf. Native restoration republishes
    // the master and half plane in its owned generation domain.
    reloaded_handle->restore(saved_coefficients);
    auto reloaded_params = trainable_params(*reloaded);
    torch::optim::Adam reloaded_optimizer(reloaded_params, torch::optim::AdamOptions(0.01));
    reloaded_optimizer.load(optimizer_input);
    torch::Tensor reload_prediction;
    {
        torch::NoGradGuard no_grad;
        reload_prediction = reloaded->forward(samples,
            std::vector<std::int64_t>(axes.begin(), axes.begin() + 8));
    }
    assert(torch::allclose(trained, reload_prediction, 1e-6, 1e-6));
    net_optimizer.zero_grad();
    reloaded_optimizer.zero_grad();
    torch::mse_loss(net->forward(samples, std::vector<std::int64_t>(axes.begin(), axes.begin() + 8)), target).backward();
    torch::mse_loss(reloaded->forward(samples, std::vector<std::int64_t>(axes.begin(), axes.begin() + 8)), target).backward();
    celleratorch::guarded_adam_step(net_optimizer, {mechanism});
    celleratorch::guarded_adam_step(reloaded_optimizer, {reloaded_handle});
    assert(torch::allclose(mechanism->coefficients(), reloaded_handle->coefficients(), 1e-6, 1e-6));
    std::remove("celleratorch-mechanism-model.pt");
    std::remove("celleratorch-mechanism-optimizer.pt");
    std::cout << "prepared mechanism autograd and guarded Adam passed\n";
    return 0;
}
