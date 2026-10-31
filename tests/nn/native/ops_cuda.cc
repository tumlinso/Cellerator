#include <Cellerator/nn/ops.hh>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <torch/autograd.h>
#include <torch/optim/sgd.h>
#include <torch/torch.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
using at::Tensor;
using cellerator::nn::ops::csr_scaled_dot_product_attention_cuda;
using cellerator::nn::ops::csr_scaled_dot_product_attention_cuda_prepared;
using cellerator::nn::ops::grouped_linear_cuda;

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
Tensor cuda_rand(std::vector<std::int64_t> shape, at::ScalarType dtype,
                 const at::Device& device, bool grad = false) {
    auto value = at::randn(shape, at::TensorOptions().dtype(dtype).device(device));
    if (grad) value.requires_grad_(true);
    return value;
}
void close(const Tensor& actual, const Tensor& expected, double atol,
           double rtol, const std::string& what) {
    check(actual.sizes() == expected.sizes(), what + ": shape mismatch");
    check(actual.scalar_type() == expected.scalar_type(), what + ": dtype mismatch");
    if (!at::allclose(actual, expected, rtol, atol, false)) {
        auto error = (actual.to(at::kFloat) - expected.to(at::kFloat)).abs();
        throw std::runtime_error(what + ": max_abs_error=" +
                                 std::to_string(error.max().item<float>()));
    }
}
std::pair<double,double> tolerance(at::ScalarType dtype) {
    return dtype == at::kHalf ? std::pair<double,double>{4e-3,4e-3}
                              : std::pair<double,double>{2e-4,2e-4};
}
Tensor grouped_reference(const Tensor& input, const Tensor& weights,
                         const std::vector<std::int64_t>& offsets) {
    auto x = input.to(at::kFloat), w = weights.to(at::kFloat);
    std::vector<Tensor> rows;
    rows.reserve(static_cast<std::size_t>(x.size(0)));
    for (std::int64_t e = 0; e + 1 < static_cast<std::int64_t>(offsets.size()); ++e) {
        for (auto r = offsets[e]; r < offsets[e + 1]; ++r)
            rows.push_back(at::matmul(w[e], x[r]));
    }
    Tensor result;
    if (rows.empty()) result = at::empty({0, w.size(1)}, x.options());
    else result = at::stack(rows, 0);
    // Preserve explicit zero derivatives for experts with no assigned rows.
    return (result + x.sum() * 0 + w.sum() * 0).to(input.scalar_type());
}
Tensor attention_reference(const Tensor& q, const Tensor& k, const Tensor& v,
                           const std::vector<std::int64_t>& offsets,
                           const std::vector<std::int64_t>& keys, double scale) {
    auto qf=q.to(at::kFloat), kf=k.to(at::kFloat), vf=v.to(at::kFloat);
    std::vector<Tensor> query_rows;
    query_rows.reserve(static_cast<std::size_t>(q.size(0)));
    for (std::int64_t qi=0; qi<q.size(0); ++qi) {
        std::vector<Tensor> heads;
        heads.reserve(static_cast<std::size_t>(q.size(1)));
        for (std::int64_t h=0; h<q.size(1); ++h) {
            if (offsets[qi] == offsets[qi+1]) {
                heads.push_back(at::zeros({v.size(2)}, qf.options()) + qf[qi][h].sum()*0 +
                                kf.sum()*0 + vf.sum()*0);
                continue;
            }
            std::vector<Tensor> logits, values;
            for (auto edge=offsets[qi]; edge<offsets[qi+1]; ++edge) {
                const auto key=keys[static_cast<std::size_t>(edge)];
                logits.push_back((qf[qi][h]*kf[key][h]).sum()*scale);
                values.push_back(vf[key][h]);
            }
            auto probabilities=at::softmax(at::stack(logits,0),0);
            heads.push_back((probabilities.unsqueeze(1)*at::stack(values,0)).sum(0));
        }
        query_rows.push_back(at::stack(heads,0));
    }
    if (query_rows.empty()) return at::empty({0,q.size(1),v.size(2)},q.options());
    return at::stack(query_rows,0).to(q.scalar_type());
}
void assert_gradients(const std::vector<Tensor>& actual,
                      const std::vector<Tensor>& reference,
                      at::ScalarType dtype, const std::string& label) {
    auto [atol,rtol]=tolerance(dtype);
    check(actual.size()==reference.size(),label+": gradient count mismatch");
    for (std::size_t i=0;i<actual.size();++i)
        close(actual[i],reference[i].to(dtype),atol,rtol,label+" gradient "+std::to_string(i));
}
void check_grouped_case(at::ScalarType dtype, const at::Device& device) {
    const std::vector<std::int64_t> offsets{0,2,2,6,7};
    auto x=cuda_rand({7,5},dtype,device,true);
    auto w=cuda_rand({4,3,5},dtype,device,true);
    auto upstream=cuda_rand({7,3},dtype,device);
    auto xr=x.detach().to(at::kFloat).requires_grad_(true);
    auto wr=w.detach().to(at::kFloat).requires_grad_(true);
    auto expected=grouped_reference(xr,wr,offsets);
    auto expected_grads=torch::autograd::grad({expected},{xr,wr},{upstream.to(at::kFloat)});
    auto actual=grouped_linear_cuda(x,w,offsets);
    auto actual_grads=torch::autograd::grad({actual},{x,w},{upstream});
    const auto [atol,rtol]=tolerance(dtype);
    close(actual,expected.to(dtype),atol,rtol,"grouped forward");
    assert_gradients(actual_grads,expected_grads,dtype,"grouped");
    check(at::equal(actual_grads[1][1],at::zeros_like(actual_grads[1][1])),
          "empty expert must receive zero gradient");
}
void check_attention_case(at::ScalarType dtype, const at::Device& device,
                          std::int64_t value_dim=2) {
    // Row 1 has no neighbors. Rows 0 and 3 contain duplicate edges.
    const std::vector<std::int64_t> offsets{0,3,3,5,7};
    const std::vector<std::int64_t> keys{0,1,1,2,0,3,3};
    constexpr double scale=0.5773502691896258;
    auto q=cuda_rand({4,2,3},dtype,device,true);
    auto k=cuda_rand({4,2,3},dtype,device,true);
    auto v=cuda_rand({4,2,value_dim},dtype,device,true);
    auto offsets_tensor=at::tensor(offsets,at::TensorOptions().dtype(at::kLong).device(device));
    auto keys_tensor=at::tensor(keys,at::TensorOptions().dtype(at::kLong).device(device));
    auto upstream=cuda_rand({4,2,value_dim},dtype,device);
    auto qr=q.detach().to(at::kFloat).requires_grad_(true);
    auto kr=k.detach().to(at::kFloat).requires_grad_(true);
    auto vr=v.detach().to(at::kFloat).requires_grad_(true);
    auto expected=attention_reference(qr,kr,vr,offsets,keys,scale);
    auto expected_grads=torch::autograd::grad({expected},{qr,kr,vr},{upstream.to(at::kFloat)});
    auto actual=csr_scaled_dot_product_attention_cuda(q,k,v,offsets_tensor,keys_tensor,scale,0.0);
    auto prepared=csr_scaled_dot_product_attention_cuda_prepared(q,k,v,offsets_tensor,keys_tensor,scale,0.0);
    auto actual_grads=torch::autograd::grad({actual},{q,k,v},{upstream});
    auto prepared_grads=torch::autograd::grad({prepared},{q,k,v},{upstream});
    const auto [atol,rtol]=tolerance(dtype);
    close(actual,expected.to(dtype),atol,rtol,"raw attention forward");
    close(prepared,expected.to(dtype),atol,rtol,"prepared attention forward");
    close(actual,prepared,atol,rtol,"raw/prepared attention parity");
    assert_gradients(actual_grads,expected_grads,dtype,"raw attention");
    assert_gradients(prepared_grads,expected_grads,dtype,"prepared attention");
    check(at::equal(actual[1],at::zeros_like(actual[1])),"empty attention row must be zero");
}
void test_grouped_forward_vjp_and_sgd(const at::Device& device) {
    for (auto dtype : {at::kFloat,at::kHalf}) check_grouped_case(dtype,device);
    auto xempty=at::empty({0,5},at::TensorOptions().dtype(at::kFloat).device(device)).requires_grad_(true);
    auto wempty=cuda_rand({3,2,5},at::kFloat,device,true);
    auto yempty=grouped_linear_cuda(xempty,wempty,{0,0,0,0});
    check(yempty.sizes()==at::IntArrayRef({0,2}),"zero-row grouped output shape mismatch");
    yempty.sum().backward();
    check(wempty.grad().defined() && at::equal(wempty.grad(),at::zeros_like(wempty)),
          "zero-row grouped operation must produce zero expert gradients");
    // Exercise ordinary leaf parameters and Torch's SGD update through native VJP.
    const std::vector<std::int64_t> offsets{0,2,4};
    auto x=cuda_rand({4,3},at::kFloat,device);
    auto w=cuda_rand({2,2,3},at::kFloat,device,true);
    auto y=grouped_linear_cuda(x,w,offsets);
    auto loss=y.square().sum(); loss.backward();
    check(w.grad().defined(),"ordinary autograd must accumulate parameter gradient");
    auto before=w.detach().clone(), gradient=w.grad().clone();
    torch::optim::SGD optimizer({w},torch::optim::SGDOptions(0.05));
    optimizer.step();
    close(w,before-0.05*gradient,1e-6,1e-6,"ordinary SGD parameter update");
}
void test_attention_forward_vjp(const at::Device& device) {
    for (auto dtype : {at::kFloat,at::kHalf})
        for (auto value_dim : {1,2,33}) check_attention_case(dtype,device,value_dim);
}
void test_empty_attention_support(const at::Device& device) {
    constexpr double scale=0.5773502691896258;
    const std::vector<std::int64_t> offsets{0,0,0,0,0};
    const std::vector<std::int64_t> keys;
    auto q=cuda_rand({4,2,3},at::kFloat,device,true);
    auto k=cuda_rand({4,2,3},at::kFloat,device,true);
    auto v=cuda_rand({4,2,1},at::kFloat,device,true);
    auto os=at::tensor(offsets,at::TensorOptions().dtype(at::kLong).device(device));
    auto ids=at::empty({0},at::TensorOptions().dtype(at::kLong).device(device));
    auto y=csr_scaled_dot_product_attention_cuda(q,k,v,os,ids,scale);
    check(at::equal(y,at::zeros_like(y)),"all-empty CSR must produce zero output");
    auto grads=torch::autograd::grad({y.sum()},{q,k,v});
    for (const auto& g:grads) check(at::equal(g,at::zeros_like(g)),"all-empty CSR must have zero VJP");
}
void test_zero_query_attention(const at::Device& device) {
    constexpr double scale=0.5773502691896258;
    auto q=cuda_rand({0,2,3},at::kFloat,device,true);
    auto k=cuda_rand({4,2,3},at::kFloat,device,true);
    auto v=cuda_rand({4,2,1},at::kFloat,device,true);
    auto os=at::zeros({1},at::TensorOptions().dtype(at::kLong).device(device));
    auto ids=at::empty({0},at::TensorOptions().dtype(at::kLong).device(device));
    auto y=csr_scaled_dot_product_attention_cuda(q,k,v,os,ids,scale);
    check(y.sizes()==at::IntArrayRef({0,2,1}),"zero-query CSR output shape mismatch");
    auto grads=torch::autograd::grad({y.sum()},{q,k,v});
    check(grads.size()==3 && grads[0].sizes()==q.sizes(),"zero-query VJP shapes mismatch");
}
void test_attention_graph_lifecycle(const at::Device& device) {
    constexpr double scale=0.5773502691896258;
    const std::vector<std::int64_t> offsets{0,1,3};
    const std::vector<std::int64_t> keys{0,1,1};
    auto os=at::tensor(offsets,at::TensorOptions().dtype(at::kLong).device(device));
    auto ids=at::tensor(keys,at::TensorOptions().dtype(at::kLong).device(device));
    auto q=cuda_rand({2,2,3},at::kFloat,device,true);
    auto k=cuda_rand({2,2,3},at::kFloat,device,true);
    auto v=cuda_rand({2,2,2},at::kFloat,device,true);
    auto loss1=csr_scaled_dot_product_attention_cuda(q,k,v,os,ids,scale).square().sum();
    auto loss2=csr_scaled_dot_product_attention_cuda(q,k,v,os,ids,scale).abs().sum();
    loss1.backward({},true);
    auto first=q.grad().clone();
    loss1.backward({},true);
    close(q.grad(),2*first,2e-4,2e-4,"attention retained graph accumulation");
    loss2.backward();
    check(at::isfinite(q.grad()).all().item<bool>(),"attention live graphs must retain finite gradients");

    auto qh=cuda_rand({2,2,3},at::kFloat,device,true);
    auto kh=cuda_rand({2,2,3},at::kFloat,device,true);
    auto vh=cuda_rand({2,2,1},at::kFloat,device,true);
    auto higher=csr_scaled_dot_product_attention_cuda(qh,kh,vh,os,ids,scale).sum();
    bool rejected=false;
    try { (void)torch::autograd::grad({higher},{qh},{},true,true); }
    catch (const c10::Error&) { rejected=true; }
    check(rejected,"attention create_graph must reject unsupported higher-order differentiation");
}
void test_attention_original_version_checks(const at::Device& device) {
    constexpr double scale=0.5773502691896258;
    auto qbase=cuda_rand({2,3,2},at::kFloat,device,true);
    auto q=qbase.transpose(1,2); // [Q,H,D], noncontiguous.
    auto k=cuda_rand({2,2,3},at::kFloat,device,true);
    auto v=cuda_rand({2,2,1},at::kFloat,device,true);
    auto os=at::tensor(std::vector<std::int64_t>{0,1,2},at::TensorOptions().dtype(at::kLong).device(device));
    auto ids=at::tensor(std::vector<std::int64_t>{0,1},at::TensorOptions().dtype(at::kLong).device(device));
    auto y=csr_scaled_dot_product_attention_cuda(q,k,v,os,ids,scale);
    { at::NoGradGuard guard; qbase.add_(1.0); }
    bool q_rejected=false;
    try { y.sum().backward(); } catch (const c10::Error&) { q_rejected=true; }
    check(q_rejected,"mutated noncontiguous original q must fail saved-tensor version check");

    auto q2=cuda_rand({2,2,3},at::kFloat,device,true);
    auto k2=cuda_rand({2,2,3},at::kFloat,device,true);
    auto v2=cuda_rand({2,2,1},at::kFloat,device,true);
    auto ids2=at::tensor(std::vector<std::int64_t>{0,1},at::TensorOptions().dtype(at::kLong).device(device));
    auto y2=csr_scaled_dot_product_attention_cuda(q2,k2,v2,os,ids2,scale);
    { at::NoGradGuard guard; ids2[0].fill_(1); }
    bool ids_rejected=false;
    try { y2.sum().backward(); } catch (const c10::Error&) { ids_rejected=true; }
    check(ids_rejected,"mutated original CSR key IDs must fail saved-tensor version check");

    auto q3=cuda_rand({2,2,3},at::kFloat,device,true);
    auto k3=cuda_rand({2,2,3},at::kFloat,device,true);
    auto v3=cuda_rand({2,2,1},at::kFloat,device,true);
    auto ids3=at::tensor(std::vector<std::int64_t>{0,1},at::TensorOptions().dtype(at::kLong).device(device));
    auto prepared=csr_scaled_dot_product_attention_cuda_prepared(q3,k3,v3,os,ids3,scale);
    { at::NoGradGuard guard; ids3[0].fill_(1); }
    bool prepared_rejected=false;
    try { prepared.sum().backward(); } catch (const c10::Error&) { prepared_rejected=true; }
    check(prepared_rejected,"mutated prepared CSR must fail saved-tensor version check");
}
void test_outstanding_and_repeated_graphs(const at::Device& device) {
    const std::vector<std::int64_t> offsets{0,1,3};
    auto w=cuda_rand({2,2,3},at::kFloat,device,true);
    auto x1=cuda_rand({3,3},at::kFloat,device,true);
    auto x2=cuda_rand({3,3},at::kFloat,device,true);
    auto loss1=grouped_linear_cuda(x1,w,offsets).square().sum();
    auto loss2=grouped_linear_cuda(x2,w,offsets).abs().sum();
    // Two live invocations own distinct executors. Retaining and replaying the
    // first graph must not reuse mutable gradient workspace from the second.
    loss1.backward({},true);
    auto first_gradient=w.grad().clone();
    loss1.backward({},true);
    close(w.grad(),2*first_gradient,2e-4,2e-4,"repeated retained graph accumulation");
    loss2.backward();
    check(at::isfinite(w.grad()).all().item<bool>(),"multiple graphs must yield finite accumulated gradients");
}
void test_first_order_boundary(const at::Device& device) {
    const std::vector<std::int64_t> offsets{0,2};
    auto x=cuda_rand({2,3},at::kFloat,device,true), w=cuda_rand({1,2,3},at::kFloat,device,true);
    auto loss=grouped_linear_cuda(x,w,offsets).square().sum();
    bool rejected=false;
    try { (void)torch::autograd::grad({loss},{x},{},true,true); }
    catch (const c10::Error&) { rejected=true; }
    check(rejected,"create_graph must reject unsupported higher-order native differentiation");
}
void test_original_noncontiguous_version_check(const at::Device& device) {
    auto base=cuda_rand({5,3},at::kFloat,device,true);
    auto x=base.transpose(0,1); // [3,5], noncontiguous input copied by the adapter.
    auto w=cuda_rand({1,2,5},at::kFloat,device,true);
    auto y=grouped_linear_cuda(x,w,{0,3});
    { at::NoGradGuard guard; base.add_(1.0); }
    bool rejected=false;
    try { y.sum().backward(); }
    catch (const c10::Error&) { rejected=true; }
    check(rejected,"mutation of original noncontiguous input must fail saved-tensor version check");

    auto x2=cuda_rand({3,5},at::kFloat,device,true);
    auto parameter_storage=cuda_rand({1,5,2},at::kFloat,device,true);
    auto noncontiguous_weight=parameter_storage.transpose(1,2); // [E,O,I].
    auto y2=grouped_linear_cuda(x2,noncontiguous_weight,{0,3});
    { at::NoGradGuard guard; parameter_storage.mul_(2.0); }
    bool weight_rejected=false;
    try { y2.sum().backward(); }
    catch (const c10::Error&) { weight_rejected=true; }
    check(weight_rejected,"mutation of original noncontiguous parameter must fail saved-tensor version check");
}
struct CallbackGate {
    std::atomic<bool> started{false};
    std::atomic<bool> open{false};
    std::atomic<bool> timed_out{false};
    int timeout_seconds=[] {
        const auto value=std::getenv("CELLERATOR_NN_TEST_GATE_TIMEOUT_SECONDS");
        return value ? std::max(3,std::min(30,std::atoi(value))) : 3;
    }();
    static void CUDART_CB wait(void* opaque) {
        auto* self=static_cast<CallbackGate*>(opaque);
        self->started.store(true,std::memory_order_release);
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(self->timeout_seconds);
        while(!self->open.load(std::memory_order_acquire) && std::chrono::steady_clock::now()<deadline)
            std::this_thread::yield();
        if(!self->open.load(std::memory_order_acquire)) {
            self->timed_out.store(true,std::memory_order_release);
            self->open.store(true,std::memory_order_release);
        }
    }
};
struct OrderedStreams {
    at::cuda::CUDAStream producer;
    at::cuda::CUDAStream consumer;
    cudaEvent_t ready=nullptr;
    CallbackGate gate;
    bool armed=false;
    explicit OrderedStreams(int device)
        : producer(at::cuda::getStreamFromPool(false,device)),
          consumer(at::cuda::getStreamFromPool(false,device)) {
        auto status=cudaEventCreateWithFlags(&ready,cudaEventDisableTiming);
        if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    }
    void arm() {
        auto status=cudaEventRecord(ready,producer.stream());
        if(status==cudaSuccess) status=cudaStreamWaitEvent(consumer.stream(),ready,0);
        if(status==cudaSuccess) status=cudaLaunchHostFunc(consumer.stream(),CallbackGate::wait,&gate);
        if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        armed=true;
    }
    void wait_until_queued() {
        if(!armed) throw std::runtime_error("stream gate was not armed");
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
        while(!gate.started.load(std::memory_order_acquire) && std::chrono::steady_clock::now()<deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if(!gate.started.load(std::memory_order_acquire))
            throw std::runtime_error("CUDA host gate did not start within five seconds");
        if(gate.timed_out.load(std::memory_order_acquire))
            throw std::runtime_error("CUDA stream gate auto-released before owner check");
    }
    void finish() noexcept {
        gate.open.store(true,std::memory_order_release);
        (void)consumer.synchronize();
        (void)producer.synchronize();
    }
    ~OrderedStreams() {
        finish();
        if(ready) (void)cudaEventDestroy(ready);
    }
};
struct FinishBeforeOwnerDestruction {
    OrderedStreams* streams;
    ~FinishBeforeOwnerDestruction() { streams->finish(); }
};
struct ExternalOwner {
    void* allocation=nullptr;
    Tensor tensor;
    std::shared_ptr<std::atomic<bool>> deleter_called=std::make_shared<std::atomic<bool>>(false);
    ExternalOwner(const std::vector<std::int64_t>& shape,const std::vector<float>& host,
                  const at::Device& device,const at::cuda::CUDAStream& upload_stream,
                  bool requires_grad) {
        const auto bytes=host.size()*sizeof(float);
        auto status=cudaMalloc(&allocation,bytes);
        if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        auto state=deleter_called;
        tensor=at::from_blob(allocation,shape,[state](void*) {
            state->store(true,std::memory_order_release);
        },at::TensorOptions().dtype(at::kFloat).device(device));
        if(requires_grad) tensor.requires_grad_(true);
        status=cudaMemcpyAsync(allocation,host.data(),bytes,cudaMemcpyHostToDevice,upload_stream.stream());
        if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    }
    void drop_tensor() { tensor=Tensor(); }
    bool released() const { return deleter_called->load(std::memory_order_acquire); }
    ~ExternalOwner() {
        drop_tensor();
        if(allocation) (void)cudaFree(allocation);
    }
};
void test_cross_stream_attention_allocator_lifetime(const at::Device& device) {
    constexpr double scale=0.7071067811865476;
    const std::vector<std::int64_t> row_offsets{0,2,2,3};
    const std::vector<std::int64_t> key_ids{0,1,2};
    auto q_cpu=at::tensor(std::vector<float>{0.2f,-0.1f,0.7f,0.3f,-0.5f,0.4f,
                                              0.6f,-0.2f,0.1f,0.8f,0.9f,-0.3f}).view({3,2,2});
    auto k_cpu=at::tensor(std::vector<float>{0.3f,0.5f,-0.4f,0.2f,0.7f,-0.6f,
                                              0.8f,0.1f,-0.2f,0.9f,0.4f,-0.5f}).view({3,2,2});
    auto v_cpu=at::tensor(std::vector<float>{0.4f,-0.3f,0.1f,0.6f,0.8f,-0.2f,
                                              -0.7f,0.5f,0.2f,0.9f,0.3f,-0.4f}).view({3,2,2});
    auto upstream_cpu=at::tensor(std::vector<float>{0.5f,-0.1f,0.2f,0.7f,-0.3f,0.4f,
                                                     0.8f,0.2f,-0.6f,0.3f,0.1f,-0.5f}).view({3,2,2});
    auto qr=q_cpu.clone().requires_grad_(true), kr=k_cpu.clone().requires_grad_(true),
         vr=v_cpu.clone().requires_grad_(true);
    auto expected=attention_reference(qr,kr,vr,row_offsets,key_ids,scale);
    auto expected_grads=torch::autograd::grad({expected},{qr,kr,vr},{upstream_cpu});

    OrderedStreams streams(device.index());
    FinishBeforeOwnerDestruction finish{&streams};
    Tensor q,k,v,offsets,keys,upstream;
    {
        c10::cuda::CUDAStreamGuard producer_guard(streams.producer);
        auto options=at::TensorOptions().dtype(at::kFloat).device(device);
        q=at::empty({3,2,2},options); q.copy_(q_cpu); q.requires_grad_(true);
        k=at::empty({3,2,2},options); k.copy_(k_cpu); k.requires_grad_(true);
        v=at::empty({3,2,2},options); v.copy_(v_cpu); v.requires_grad_(true);
        upstream=at::empty({3,2,2},options); upstream.copy_(upstream_cpu);
        offsets=at::empty({4},options.dtype(at::kLong));
        offsets.copy_(at::tensor(row_offsets,at::TensorOptions().dtype(at::kLong)));
        keys=at::empty({3},options.dtype(at::kLong));
        keys.copy_(at::tensor(key_ids,at::TensorOptions().dtype(at::kLong)));
    }
    streams.arm();
    Tensor output_for_check;
    std::vector<Tensor> actual_grads;
    {
        c10::cuda::CUDAStreamGuard consumer_guard(streams.consumer);
        auto output=csr_scaled_dot_product_attention_cuda_prepared(q,k,v,offsets,keys,scale);
        actual_grads=torch::autograd::grad({output},{q,k,v},{upstream});
        output_for_check=output.detach();
    }
    // Backward has returned while its CUDA work remains behind the host gate.
    // Drop the output graph and inputs so their blocks can be recycled on the
    // creation stream if recordStream ownership is missing.
    q=Tensor(); k=Tensor(); v=Tensor(); offsets=Tensor(); keys=Tensor(); upstream=Tensor();
    streams.wait_until_queued();
    std::vector<Tensor> churn;
    {
        c10::cuda::CUDAStreamGuard producer_guard(streams.producer);
        auto options=at::TensorOptions().dtype(at::kFloat).device(device);
        for(int i=0;i<24;++i) {
            churn.push_back(at::empty({3,2,2},options).zero_());
            churn.push_back(at::empty({4},options.dtype(at::kLong)).zero_());
            churn.push_back(at::empty({3},options.dtype(at::kLong)).zero_());
        }
    }
    streams.producer.synchronize();
    {
        c10::cuda::CUDAStreamGuard producer_guard(streams.producer);
        churn.clear();
    }
    streams.finish();
    close(output_for_check.to(at::kCPU),expected,2e-4,2e-4,"cross-stream prepared attention output");
    for(std::size_t i=0;i<actual_grads.size();++i)
        close(actual_grads[i].to(at::kCPU),expected_grads[i],2e-4,2e-4,
              "cross-stream attention gradient "+std::to_string(i));
}

void expect_release_after_stream(const std::shared_ptr<std::atomic<bool>>& called,
                                 const char* description) {
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    while(!called->load(std::memory_order_acquire) && std::chrono::steady_clock::now()<deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    check(called->load(std::memory_order_acquire),std::string(description)+
          " deleter did not run within five seconds of stream completion");
}
void test_external_storage_completion_retention(const at::Device& device) {
    const std::vector<float> host_x{1.0f,2.0f,3.0f,4.0f,5.0f,6.0f};
    const std::vector<float> host_w{1.0f,0.5f,-1.0f,2.0f,0.25f,-0.5f,1.5f,1.0f};
    auto expected=grouped_reference(at::tensor(host_x).view({3,2}),
                                    at::tensor(host_w).view({2,2,2}),{0,1,3});
    {
        OrderedStreams streams(device.index());
        ExternalOwner x({3,2},host_x,device,streams.producer,false);
        ExternalOwner w({2,2,2},host_w,device,streams.producer,false);
        FinishBeforeOwnerDestruction finish{&streams};
        streams.arm();
        Tensor output;
        {
            c10::cuda::CUDAStreamGuard stream_guard(streams.consumer);
            at::InferenceMode inference;
            output=grouped_linear_cuda(x.tensor,w.tensor,{0,1,3});
            check(!output.requires_grad(),"inference output unexpectedly tracks autograd");
            x.drop_tensor(); w.drop_tensor();
        }
        streams.wait_until_queued();
        check(!x.released() && !w.released(),
              "inference from_blob owners must remain live while consumer stream is gated");
        streams.finish();
        expect_release_after_stream(x.deleter_called,"inference input");
        expect_release_after_stream(w.deleter_called,"inference weight");
        close(output.to(at::kCPU),expected,2e-4,2e-4,"external inference output");
    }
    {
        OrderedStreams streams(device.index());
        ExternalOwner x({3,2},host_x,device,streams.producer,true);
        ExternalOwner w({2,2,2},host_w,device,streams.producer,true);
        FinishBeforeOwnerDestruction finish{&streams};
        streams.arm();
        {
            c10::cuda::CUDAStreamGuard stream_guard(streams.consumer);
            auto output=grouped_linear_cuda(x.tensor,w.tensor,{0,1,3});
            check(output.requires_grad(),"training output must retain its graph");
            output=Tensor(); // Destroy the graph while its forward remains behind the gate.
            x.drop_tensor(); w.drop_tensor();
        }
        streams.wait_until_queued();
        check(!x.released() && !w.released(),
              "early-disposed graph owners must remain live while consumer stream is gated");
        streams.finish();
        expect_release_after_stream(x.deleter_called,"early-disposed input");
        expect_release_after_stream(w.deleter_called,"early-disposed weight");
    }
}

void test_nondefault_stream_and_dropped_handles(const at::Device& device) {
    auto stream=at::cuda::getStreamFromPool(false,device.index());
    {
        c10::cuda::CUDAStreamGuard stream_guard(stream);
        auto source=cuda_rand({6,4},at::kFloat,device,true);
        auto input=source.slice(0,0,6,1);
        auto weight=cuda_rand({2,3,4},at::kFloat,device,true);
        auto output=grouped_linear_cuda(input,weight,{0,2,6});
        auto loss=output.square().sum();
        // Drop caller handles before backward and churn the allocator. Saved
        // original and contiguous tensors plus allocator stream records must
        // keep all borrowed storage alive on this non-default stream.
        input=Tensor(); weight=Tensor(); output=Tensor();
        std::vector<Tensor> churn;
        for (int i=0;i<32;++i) churn.push_back(at::empty({4096},at::TensorOptions().device(device)));
        churn.clear();
        loss.backward();
        check(source.grad().defined() && at::isfinite(source.grad()).all().item<bool>(),
              "non-default stream backward after dropping handles must produce finite gradient");
    }
    // Exercise pending forwards whose autograd graphs are immediately dropped.
    // Saved allocations and native execution state must survive until queued
    // kernels finish even though no backward will consume the result.
    {
        c10::cuda::CUDAStreamGuard stream_guard(stream);
        auto x=cuda_rand({8,4},at::kFloat,device);
        auto w=cuda_rand({2,3,4},at::kFloat,device);
        auto y=grouped_linear_cuda(x,w,{0,3,8});
        y=Tensor(); x=Tensor(); w=Tensor();
        auto q=cuda_rand({4,2,3},at::kFloat,device);
        auto k=cuda_rand({4,2,3},at::kFloat,device);
        auto v=cuda_rand({4,2,1},at::kFloat,device);
        auto os=at::tensor(std::vector<std::int64_t>{0,2,3,3,4},at::TensorOptions().dtype(at::kLong).device(device));
        auto ids=at::tensor(std::vector<std::int64_t>{0,1,2,3},at::TensorOptions().dtype(at::kLong).device(device));
        auto out=csr_scaled_dot_product_attention_cuda(q,k,v,os,ids,0.5773502691896258);
        out=Tensor(); q=Tensor(); k=Tensor(); v=Tensor(); os=Tensor(); ids=Tensor();
        std::vector<Tensor> churn;
        for(int i=0;i<32;++i) churn.push_back(at::empty({4096},at::TensorOptions().device(device)));
        churn.clear();
    }
    stream.synchronize();
}
template<class F> void run_case(const char* name,F&& fn,int& passed,int& failed) {
    try { fn(); ++passed; std::cout<<"PASS "<<name<<"\n"; }
    catch (const std::exception& e) { ++failed; std::cerr<<"FAIL "<<name<<": "<<e.what()<<"\n"; }
}
void run_device_case(bool sanitizer_mode) {
    torch::manual_seed(0xCE11);
    const auto device=at::Device(at::kCUDA,c10::cuda::current_device());
    cudaDeviceProp properties{};
    auto status=cudaGetDeviceProperties(&properties,device.index());
    if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    char uuid[80]{};
    auto byte = [&](int i) { return static_cast<unsigned>(
        static_cast<unsigned char>(properties.uuid.bytes[i])); };
    std::snprintf(uuid,sizeof(uuid),"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        byte(0),byte(1),byte(2),byte(3),byte(4),byte(5),byte(6),byte(7),
        byte(8),byte(9),byte(10),byte(11),byte(12),byte(13),byte(14),byte(15));
    std::cout<<"Device: "<<properties.name<<" UUID "<<uuid<<" compute "
             <<properties.major<<"."<<properties.minor<<"\n";
    int passed=0,failed=0;
    run_case("grouped FP16/FP32 forward, VJP, empty expert, SGD",[&]{test_grouped_forward_vjp_and_sgd(device);},passed,failed);
    run_case("staged CSR attention FP16/FP32 forward and q/k/v VJP at V=1,2,33",[&]{test_attention_forward_vjp(device);},passed,failed);
    run_case("empty CSR neighborhoods and zero-query tensors",[&]{test_empty_attention_support(device);test_zero_query_attention(device);},passed,failed);
    run_case("attention outstanding/repeated graphs and first-order boundary",[&]{test_attention_graph_lifecycle(device);},passed,failed);
    run_case("attention original tensor and CSR version checks",[&]{test_attention_original_version_checks(device);},passed,failed);
    run_case("two outstanding grouped graphs and retained repeated backward",[&]{test_outstanding_and_repeated_graphs(device);},passed,failed);
    run_case("explicit first-order boundary",[&]{test_first_order_boundary(device);},passed,failed);
    run_case("original noncontiguous tensor version check",[&]{test_original_noncontiguous_version_check(device);},passed,failed);
    run_case("non-default stream, dropped handles and allocator churn",[&]{test_nondefault_stream_and_dropped_handles(device);},passed,failed);
    if (!sanitizer_mode) {
        run_case("cross-stream attention allocator ownership and VJP",[&]{test_cross_stream_attention_allocator_lifetime(device);},passed,failed);
        run_case("external from_blob inference and disposed-graph event retention",[&]{test_external_storage_completion_retention(device);},passed,failed);
    } else {
        std::cout << "SKIP two host-gated async-submission fixtures under sanitizer launch instrumentation; required in normal qualification\n";
    }
    auto sync=cudaDeviceSynchronize();
    if(sync!=cudaSuccess) { ++failed; std::cerr<<"FAIL final device synchronization: "<<cudaGetErrorString(sync)<<"\n"; }
    std::cout<<"CUDA adapter qualification: "<<passed<<" passed, "<<failed<<" failed, "<<(sanitizer_mode ? 2 : 0)<<" skipped\n";
    if(failed) throw std::runtime_error("one or more CUDA qualification cases failed");
}
} // namespace

int main(int argc,char** argv) {
    if(argc!=2 || (std::string(argv[1])!="--device" && std::string(argv[1])!="--memcheck")) {
        std::cout<<"CUDA NN adapter qualification is opt-in. Run with --device inside an admitted Device Scheduler job.\n";
        return 0;
    }
    try { run_device_case(std::string(argv[1])=="--memcheck"); return 0; }
    catch(const std::exception& e) { std::cerr<<"CUDA adapter qualification failed: "<<e.what()<<"\n"; return 1; }
}
