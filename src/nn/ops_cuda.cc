#include <Cellerator/nn/ops.hh>
#include <Cellerator/modeling/cuda/attention.hh>
#include <Cellerator/modeling/cuda/factorized_interaction.hh>
#include <Cellerator/modeling/cuda/grouped_linear.hh>
#include <Cellerator/memory/cuda/resources.hh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/autograd/custom_function.h>
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <cstdint>
#include <limits>
#include <utility>
#include <thread>
#include <memory>
#include <mutex>

namespace cellerator::nn::ops {
namespace {
namespace native = cellerator::modeling::cuda;
namespace resources = cellerator::memory::cuda;
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

void payload(const at::Tensor& t, const char* name) {
    const auto backend = c10::cuda::CUDACachingAllocator::name();
    TORCH_CHECK(backend == "native" || backend == "cudaMallocAsync",
                "native NN adapters require the Torch native or cudaMallocAsync allocator backend");
    TORCH_CHECK(t.is_cuda() && t.layout() == at::kStrided &&
                (t.scalar_type() == at::kHalf || t.scalar_type() == at::kFloat),
                name, " must be a strided CUDA FP16 or FP32 tensor");
}
void matching(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.device() == b.device() && a.scalar_type() == b.scalar_type(),
                "native operands must have matching device and dtype");
}
bool stream_managed(const at::Tensor& t) {
    const auto allocator = c10::cuda::CUDACachingAllocator::get();
    const auto storage = t.storage();
    const auto deleter = allocator->raw_deleter();
    return storage.allocator() == allocator &&
           (!deleter || storage.data_ptr().get_deleter() == deleter);
}
void record(const at::Tensor& t, c10::cuda::CUDAStream stream) {
    if (t.defined() && t.is_cuda() && t.numel() && stream_managed(t))
        c10::cuda::CUDACachingAllocator::recordStream(t.storage().data_ptr(), stream);
}
void record(const variable_list& tensors, c10::cuda::CUDAStream stream) {
    for (const auto& t : tensors) record(t, stream);
}
// recordStream only manages allocations belonging to the active Torch
// allocator. Foreign from_blob/DLPack storage needs its actual Tensor owner
// retained through a completion event, including no-grad and discarded graphs.
// This private queue releases owners on a host worker, never a CUDA callback.
class ExternalCompletions {
    struct Pending {
        cudaEvent_t event; int device; variable_list owners;
        c10::intrusive_ptr<torch::CustomClassHolder> resource;
    };
    std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<Pending> pending_;
    bool stopping_ = false;
    std::thread worker_;
    void run() noexcept {
        for (;;) {
            Pending next;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock,[&]{ return stopping_ || !pending_.empty(); });
                if (pending_.empty()) return;
                next = std::move(pending_.front()); pending_.pop_front();
            }
            // CUDA failures are reported by the submitting operation/stream.
            // Even a failed completion wait must not throw on this cleanup path.
            (void)cudaSetDevice(next.device);
            (void)cudaEventSynchronize(next.event);
            (void)cudaEventDestroy(next.event);
            next.owners.clear();
            next.resource.reset();
        }
    }
public:
    ExternalCompletions() : worker_([this]{ run(); }) {}
    ~ExternalCompletions() {
        { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
        condition_.notify_one();
        worker_.join(); // Drain recorded work before process/library teardown.
    }
    void add(cudaEvent_t event,int device,const variable_list& owners,
             const c10::intrusive_ptr<torch::CustomClassHolder>& resource) {
        { std::lock_guard<std::mutex> lock(mutex_); pending_.push_back({event,device,owners,resource}); }
        condition_.notify_one();
    }
    static ExternalCompletions& instance() { static ExternalCompletions value; return value; }
};
class ExternalBorrowHold {
    variable_list owners_;
    c10::intrusive_ptr<torch::CustomClassHolder> resource_;
    c10::cuda::CUDAStream stream_;
public:
    ExternalBorrowHold(const variable_list& tensors,c10::cuda::CUDAStream stream) : stream_(stream) {
        for (const auto& t : tensors) {
            if (t.defined() && t.is_cuda() && t.numel() && !stream_managed(t)) owners_.push_back(t);
        }
    }
    void resource(c10::intrusive_ptr<torch::CustomClassHolder> value) { resource_ = std::move(value); }
    ~ExternalBorrowHold() noexcept {
        if (owners_.empty() && !resource_) return;
        cudaEvent_t event = nullptr;
        if (cudaEventCreateWithFlags(&event,cudaEventDisableTiming) == cudaSuccess &&
            cudaEventRecord(event,stream_.stream()) == cudaSuccess) {
            try { ExternalCompletions::instance().add(event,stream_.device_index(),owners_,resource_); return; }
            catch (...) { /* Low-memory/thread failure: synchronous safe fallback. */ }
        }
        (void)cudaStreamSynchronize(stream_.stream());
        if (event) (void)cudaEventDestroy(event);
    }
};
resources::ConstDeviceView view(const at::Tensor& t) {
    return {t.const_data_ptr(), static_cast<std::size_t>(t.nbytes()), t.get_device()};
}
resources::DeviceView writable(const at::Tensor& t) {
    return {t.data_ptr(), static_cast<std::size_t>(t.nbytes()), t.get_device()};
}
struct GroupState : torch::CustomClassHolder {
    std::unique_ptr<native::GroupedLinear> executor;
    std::mutex mutex;
};
class GroupedFunction : public torch::autograd::Function<GroupedFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor input,
                              at::Tensor weights, std::vector<std::int64_t> offsets) {
        c10::cuda::CUDAGuard guard(input.device());
        const auto stream = c10::cuda::getCurrentCUDAStream(input.get_device());
        ExternalBorrowHold hold({input,weights},stream);
        record({input,weights},stream); // Protect copies even if later preparation throws.
        auto x = input.contiguous(), w = weights.contiguous();
        auto y = at::empty({x.size(0), w.size(1)}, x.options());
        auto state = c10::make_intrusive<GroupState>();
        // Handle destruction can synchronize; defer it past queued use even
        // when no-grad discards the Function context immediately.
        hold.resource(state);
        std::vector<resources::ConstDeviceView> views;
        for (std::int64_t e = 0; e < w.size(0); ++e) views.push_back(view(w[e]));
        state->executor = std::make_unique<native::GroupedLinear>(
            x.size(1), w.size(1), std::move(offsets), std::move(views),
            x.scalar_type() == at::kHalf ? resources::DType::float16 : resources::DType::float32,
            x.get_device());
        record({input, weights, x, w, y}, stream);
        state->executor->forward(view(x), writable(y), resources::Stream::borrow(stream.stream(), x.get_device()));
        ctx->save_for_backward({input, weights, x, w});
        ctx->saved_data["state"] = c10::IValue::make_capsule(std::move(state));
        return y;
    }
    static variable_list backward(AutogradContext* ctx, variable_list gradients) {
        TORCH_CHECK(!at::GradMode::is_enabled(), "native grouped linear supports first-order autograd only");
        auto saved = ctx->get_saved_variables(); // Includes original tensors: version checks precede pointer use.
        const auto& x = saved[2]; const auto& w = saved[3];
        c10::cuda::CUDAGuard guard(x.device());
        const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
        auto owners = saved; owners.push_back(gradients[0]);
        ExternalBorrowHold hold(owners,stream);
        record(owners,stream);
        auto dy = gradients[0].contiguous();
        auto dx = at::empty_like(x), dw = at::zeros_like(w);
        auto state = c10::static_intrusive_pointer_cast<GroupState>(ctx->saved_data["state"].toCapsule());
        hold.resource(state);
        std::lock_guard<std::mutex> lock(state->mutex);
        std::vector<resources::DeviceView> views;
        for (std::int64_t e = 0; e < w.size(0); ++e) views.push_back(writable(dw[e]));
        record(saved, stream); record({gradients[0], dy, dx, dw}, stream);
        state->executor->backward(view(x), view(dy), writable(dx), views,
            native::GradientMode::overwrite, native::GradientMode::overwrite,
            resources::Stream::borrow(stream.stream(), x.get_device()));
        return {dx, dw, at::Tensor()};
    }
};

// Storage is [H,Q,D] for each native 2D per-head call. Each head has private
// intermediates; no workspace is shared between autograd invocations.
native::SparseAttentionWorkspace workspace(const variable_list& s, std::int64_t h) {
    auto ptr = [h](const at::Tensor& t) { return t[h].data_ptr<float>(); };
    return {ptr(s[10]), static_cast<std::size_t>(s[10].size(1)),
            ptr(s[11]), static_cast<std::size_t>(s[11].size(1)),
            ptr(s[12]), static_cast<std::size_t>(s[12].size(1)),
            ptr(s[13]), static_cast<std::size_t>(s[13].size(1))};
}
native::SparseAttentionPlan plan(const variable_list& s) {
    return {s[8].data_ptr<std::int64_t>(), s[9].data_ptr<std::int64_t>(),
            s[5].size(1), s[6].size(1), s[9].numel(), s[5].size(2), s[7].size(2)};
}
void attention_forward(const variable_list& s, at::Tensor& output, float scale,
                        cudaStream_t stream) {
    auto p = plan(s);
    for (std::int64_t h = 0; h < s[5].size(0); ++h) {
        const auto q = s[5][h], k = s[6][h], v = s[7][h], y = output[h];
        cudaError_t status;
        if (q.scalar_type() == at::kHalf)
            status = native::sparse_attention_fp16(
                reinterpret_cast<const __half*>(q.const_data_ptr()), q.numel(),
                reinterpret_cast<const __half*>(k.const_data_ptr()), k.numel(),
                reinterpret_cast<const __half*>(v.const_data_ptr()), v.numel(),
                reinterpret_cast<__half*>(y.data_ptr()), y.numel(), p, scale, workspace(s,h), stream);
        else status = native::sparse_attention_fp32(q.data_ptr<float>(), q.numel(),
                k.data_ptr<float>(), k.numel(), v.data_ptr<float>(), v.numel(),
                y.data_ptr<float>(), y.numel(), p, scale, workspace(s,h), stream);
        resources::check(status, "native staged CSR attention forward");
    }
}
class AttentionFunction : public torch::autograd::Function<AttentionFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor q, at::Tensor k,
                              at::Tensor v, at::Tensor offsets, at::Tensor keys, double scale, bool prepared) {
        c10::cuda::CUDAGuard guard(q.device());
        const auto stream = c10::cuda::getCurrentCUDAStream(q.get_device());
        ExternalBorrowHold hold({q,k,v,offsets,keys},stream);
        record({q,k,v,offsets,keys},stream);
        auto os = prepared ? offsets : offsets.contiguous().clone();
        auto ks = prepared ? keys : keys.contiguous().clone();
        if (!prepared) {
        // Raw structure admission is synchronous and precedes any native kernel.
        const auto oh = os.to(at::kCPU), kh = ks.to(at::kCPU);
        auto o = oh.data_ptr<std::int64_t>(); auto ids = kh.data_ptr<std::int64_t>();
        TORCH_CHECK(o[0] == 0 && o[q.size(0)] == keys.numel(), "CSR offsets must span all edges");
        for (std::int64_t i = 0; i < q.size(0); ++i)
            TORCH_CHECK(o[i] >= 0 && o[i] <= o[i+1], "CSR offsets must be nonnegative and monotone");
        for (std::int64_t i = 0; i < keys.numel(); ++i)
            TORCH_CHECK(ids[i] >= 0 && ids[i] < k.size(0), "CSR key index is out of bounds");
        }
        const auto h = q.size(1), edges = keys.numel();
        auto floats = q.options().dtype(at::kFloat);
        variable_list s{q,k,v,offsets,keys,
            q.permute({1,0,2}).contiguous(), k.permute({1,0,2}).contiguous(),
            v.permute({1,0,2}).contiguous(), os,ks,
            at::empty({h,edges},floats), at::empty({h,edges},floats),
            at::empty({h,edges},floats), at::empty({h,q.size(0)},floats)};
        auto packed_output = at::empty({h,q.size(0),v.size(2)}, q.options());
        record(s,stream); record(packed_output,stream);
        attention_forward(s,packed_output,static_cast<float>(scale),stream.stream());
        auto output = packed_output.permute({1,0,2}).contiguous();
        record(output,stream);
        ctx->save_for_backward(s); ctx->saved_data["scale"] = scale;
        return output;
    }
    static variable_list backward(AutogradContext* ctx, variable_list gradients) {
        TORCH_CHECK(!at::GradMode::is_enabled(), "native CSR attention supports first-order autograd only");
        auto s = ctx->get_saved_variables();
        c10::cuda::CUDAGuard guard(s[0].device());
        const auto stream = c10::cuda::getCurrentCUDAStream(s[0].get_device());
        auto owners = s; owners.push_back(gradients[0]);
        ExternalBorrowHold hold(owners,stream);
        record(owners,stream);
        auto dy = gradients[0].permute({1,0,2}).contiguous();
        auto dq = at::empty_like(s[5],s[5].options().dtype(at::kFloat));
        auto dk = at::empty_like(s[6],s[6].options().dtype(at::kFloat));
        auto dv = at::empty_like(s[7],s[7].options().dtype(at::kFloat));
        // The forward probabilities are read-only. Backward scratch is private
        // per backward call, allowing repeat/concurrent backward of one graph.
        s[12] = at::empty_like(s[12]); s[13] = at::empty_like(s[13]);
        record(s,stream); record({gradients[0],dy,dq,dk,dv},stream);
        auto p = plan(s); const float scale = static_cast<float>(ctx->saved_data["scale"].toDouble());
        for (std::int64_t h = 0; h < s[5].size(0); ++h) {
            const auto q=s[5][h], k=s[6][h], v=s[7][h], g=dy[h];
            cudaError_t status;
            if (q.scalar_type() == at::kHalf)
                status = native::sparse_attention_backward_fp16(
                    reinterpret_cast<const __half*>(q.const_data_ptr()), q.numel(),
                    reinterpret_cast<const __half*>(k.const_data_ptr()), k.numel(),
                    reinterpret_cast<const __half*>(v.const_data_ptr()), v.numel(),
                    reinterpret_cast<const __half*>(g.const_data_ptr()), g.numel(),p,scale,workspace(s,h),
                    dq[h].data_ptr<float>(),dq[h].numel(),dk[h].data_ptr<float>(),dk[h].numel(),
                    dv[h].data_ptr<float>(),dv[h].numel(),native::GradientWriteMode::zero,stream.stream());
            else status = native::sparse_attention_backward_fp32(q.data_ptr<float>(),q.numel(),
                    k.data_ptr<float>(),k.numel(),v.data_ptr<float>(),v.numel(),g.data_ptr<float>(),g.numel(),
                    p,scale,workspace(s,h),dq[h].data_ptr<float>(),dq[h].numel(),
                    dk[h].data_ptr<float>(),dk[h].numel(),dv[h].data_ptr<float>(),dv[h].numel(),
                    native::GradientWriteMode::zero,stream.stream());
            resources::check(status,"native staged CSR attention backward");
        }
        auto restore = [&](const at::Tensor& t) { return t.permute({1,0,2}).to(s[0].scalar_type()).contiguous(); };
        auto gq=restore(dq), gk=restore(dk), gv=restore(dv);
        record({gq,gk,gv},stream);
        return {gq,gk,gv,at::Tensor(),at::Tensor(),at::Tensor(),at::Tensor()};
    }
};

resources::DType factorized_dtype(at::ScalarType type) {
    TORCH_CHECK(type == at::kHalf || type == at::kFloat,
                "factorized interaction supports FP16 or FP32 storage");
    return type == at::kHalf ? resources::DType::float16 : resources::DType::float32;
}

at::Tensor float_workspace(std::size_t bytes, const at::TensorOptions& options) {
    const auto floats = bytes / sizeof(float) + (bytes % sizeof(float) != 0);
    TORCH_CHECK(floats <= static_cast<std::size_t>(
                    std::numeric_limits<std::int64_t>::max()),
                "factorized interaction workspace exceeds tensor extent");
    return at::empty({static_cast<std::int64_t>(floats)}, options.dtype(at::kFloat));
}

void validate_factorized_inputs(
    const layout::FactorizedInteractionView& plan,
    const at::Tensor& x, const at::Tensor& z, const at::Tensor& alpha,
    const at::Tensor& bias, const at::Tensor& gamma,
    at::ScalarType output_dtype) {
    (void)factorized_dtype(output_dtype);
    for (const auto* item : {&x, &z, &alpha, &bias, &gamma}) {
        payload(*item, "factorized interaction operand");
        TORCH_CHECK(item->is_contiguous(),
                    "factorized interaction operands must be contiguous");
        TORCH_CHECK(item->device() == x.device(),
                    "factorized interaction operands must share one CUDA device");
    }
    TORCH_CHECK(x.dim() == 3 && z.dim() == 2 && alpha.dim() == 2 &&
                bias.dim() == 1 && gamma.dim() == 1,
                "factorized interaction requires x[O,D,C], z[Q,C], alpha[A,D], bias[B], gamma[G]");
    TORCH_CHECK(x.scalar_type() == z.scalar_type(),
                "factorized activations and context must use one dtype");
    TORCH_CHECK(alpha.scalar_type() == bias.scalar_type() &&
                alpha.scalar_type() == gamma.scalar_type(),
                "factorized parameter tensors must use one dtype");
    TORCH_CHECK(plan.extents.outputs <= static_cast<std::uint64_t>(
                    std::numeric_limits<std::int64_t>::max()),
                "factorized output extent exceeds Torch dimension range");
    TORCH_CHECK(static_cast<std::uint64_t>(x.size(0)) == plan.extents.inputs &&
                static_cast<std::uint64_t>(z.size(0)) == plan.extents.contexts &&
                static_cast<std::uint64_t>(alpha.size(0)) == plan.extents.alpha_rows &&
                static_cast<std::uint64_t>(bias.size(0)) == plan.extents.biases &&
                static_cast<std::uint64_t>(gamma.size(0)) == plan.extents.gammas &&
                z.size(1) == x.size(2) && alpha.size(1) == x.size(1),
                "factorized tensor shapes differ from the prepared layout");
}

struct FactorizedAutogradState : torch::CustomClassHolder {
    FactorizedAutogradState(
                            std::shared_ptr<native::FactorizedInteractionDeviceLayout> selected_layout,
                            native::FactorizedInteractionProblem shape,
                            native::FactorizedInteractionPrecision selected_precision,
                            native::FactorizedInteractionRoute selected_route,
                            const resources::Stream& stream)
        : device(stream.device()), forward_stream(stream.get()), problem(shape),
          precision(selected_precision), route(selected_route),
          device_layout(std::move(selected_layout)), forward_complete(device, false) {
        TORCH_CHECK(device_layout != nullptr,
                    "factorized interaction requires a prepared device layout");
        TORCH_CHECK(device_layout->device() == device,
                    "factorized device layout and tensors must use one CUDA device");
    }

    int device;
    cudaStream_t forward_stream;
    native::FactorizedInteractionProblem problem;
    native::FactorizedInteractionPrecision precision;
    native::FactorizedInteractionRoute route;
    std::shared_ptr<native::FactorizedInteractionDeviceLayout> device_layout;
    native::FactorizedInteractionWorkspace workspace;
    resources::Event forward_complete;
};

class FactorizedInteractionFunction : public torch::autograd::Function<FactorizedInteractionFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx,
                              std::shared_ptr<native::FactorizedInteractionDeviceLayout> device_layout,
                              at::Tensor x, at::Tensor z, at::Tensor alpha,
                              at::Tensor bias, at::Tensor gamma,
                              std::int64_t output_type, std::int64_t route_code) {
        TORCH_CHECK(device_layout != nullptr,
                    "factorized interaction requires a prepared device layout");
        const auto output_dtype = static_cast<at::ScalarType>(output_type);
        TORCH_CHECK(route_code == 0 || route_code == 1,
                    "factorized interaction route must be packet or direct");
        const auto& plan = device_layout->view();
        validate_factorized_inputs(plan, x, z, alpha, bias, gamma, output_dtype);
        c10::cuda::CUDAGuard guard(x.device());
        const auto torch_stream = c10::cuda::getCurrentCUDAStream(x.get_device());
        ExternalBorrowHold hold({x, z, alpha, bias, gamma}, torch_stream);
        record({x, z, alpha, bias, gamma}, torch_stream);
        const auto native_stream = resources::Stream::borrow(
            torch_stream.stream(), x.get_device());
        const native::FactorizedInteractionProblem problem{
            static_cast<std::uint64_t>(x.size(2)),
            static_cast<std::uint64_t>(x.size(1))};
        const native::FactorizedInteractionPrecision precision{
            factorized_dtype(x.scalar_type()), factorized_dtype(alpha.scalar_type()),
            factorized_dtype(output_dtype)};
        auto state = c10::make_intrusive<FactorizedAutogradState>(
            std::move(device_layout), problem, precision,
            route_code == 0 ? native::FactorizedInteractionRoute::packet
                            : native::FactorizedInteractionRoute::direct,
            native_stream);
        hold.resource(state);
        state->device_layout->wait_on(native_stream);
        resources::check(native::factorized_interaction_workspace(
            state->device_layout->view(), problem, state->route, &state->workspace),
            "factorized interaction workspace query");

        auto output = at::empty(
            {static_cast<std::int64_t>(plan.extents.outputs), x.size(1), x.size(2)},
            x.options().dtype(output_dtype));
        auto tape = float_workspace(state->workspace.tape_bytes, x.options());
        auto scratch = float_workspace(state->workspace.forward_bytes, x.options());
        record({output, tape, scratch}, torch_stream);
        const native::FactorizedInteractionValues values{
            view(x), view(z), view(alpha), view(bias), view(gamma)};
        resources::check(native::factorized_interaction_forward(
            state->device_layout->view(), problem, precision, values,
            writable(output), writable(tape), writable(scratch),
            torch_stream.stream(), state->route), "factorized interaction forward");
        state->forward_complete.record(native_stream);
        ctx->save_for_backward({x, z, alpha, bias, gamma, tape});
        ctx->saved_data["state"] = c10::IValue::make_capsule(std::move(state));
        return output;
    }

    static variable_list backward(AutogradContext* ctx, variable_list gradients) {
        TORCH_CHECK(!at::GradMode::is_enabled(),
                    "factorized interaction supports first-order autograd only");
        TORCH_CHECK(gradients.size() == 1 && gradients[0].defined(),
                    "factorized interaction requires an output cotangent");
        const auto saved = ctx->get_saved_variables();
        const auto& x = saved[0]; const auto& z = saved[1];
        const auto& alpha = saved[2]; const auto& bias = saved[3];
        const auto& gamma = saved[4]; const auto& tape = saved[5];
        c10::cuda::CUDAGuard guard(x.device());
        const auto torch_stream = c10::cuda::getCurrentCUDAStream(x.get_device());
        variable_list owners = saved;
        owners.push_back(gradients[0]);
        ExternalBorrowHold hold(owners, torch_stream);
        record(owners, torch_stream);
        auto dy = gradients[0].to(at::kFloat).contiguous();
        auto state = c10::static_intrusive_pointer_cast<FactorizedAutogradState>(
            ctx->saved_data["state"].toCapsule());
        hold.resource(state);
        auto native_stream = resources::Stream::borrow(
            torch_stream.stream(), x.get_device());
        if (torch_stream.stream() != state->forward_stream)
            native_stream.wait(state->forward_complete);
        state->device_layout->wait_on(native_stream);

        auto dx32 = at::empty(x.sizes(), x.options().dtype(at::kFloat));
        auto dz32 = at::empty(z.sizes(), z.options().dtype(at::kFloat));
        auto dalpha32 = at::empty(alpha.sizes(), alpha.options().dtype(at::kFloat));
        auto dbias32 = at::empty(bias.sizes(), bias.options().dtype(at::kFloat));
        auto dgamma32 = at::empty(gamma.sizes(), gamma.options().dtype(at::kFloat));
        auto scratch = float_workspace(state->workspace.backward_bytes, x.options());
        record({gradients[0], dy, dx32, dz32, dalpha32, dbias32, dgamma32, scratch},
               torch_stream);
        const native::FactorizedInteractionValues values{
            view(x), view(z), view(alpha), view(bias), view(gamma)};
        const native::FactorizedInteractionGradients destinations{
            writable(dx32), writable(dz32), writable(dalpha32),
            writable(dbias32), writable(dgamma32)};
        resources::check(native::factorized_interaction_backward(
            state->device_layout->view(), state->problem, state->precision, values,
            view(dy), view(tape), destinations, writable(scratch),
            native::FactorizedInteractionGradientMode::overwrite,
            torch_stream.stream(), state->route), "factorized interaction backward");
        auto dx = dx32.to(x.scalar_type());
        auto dz = dz32.to(z.scalar_type());
        auto dalpha = dalpha32.to(alpha.scalar_type());
        auto dbias = dbias32.to(bias.scalar_type());
        auto dgamma = dgamma32.to(gamma.scalar_type());
        record({dx, dz, dalpha, dbias, dgamma}, torch_stream);
        return {at::Tensor(), dx, dz, dalpha, dbias, dgamma,
                at::Tensor(), at::Tensor()};
    }
};
} // namespace

at::Tensor grouped_linear_cuda(const at::Tensor& input, const at::Tensor& weights,
                               const std::vector<std::int64_t>& offsets) {
    payload(input,"input"); payload(weights,"weights"); matching(input,weights);
    TORCH_CHECK(input.dim()==2 && weights.dim()==3 && input.size(1)==weights.size(2),
                "grouped linear requires input[R,I] and weights[E,O,I]");
    TORCH_CHECK(offsets.size()==static_cast<std::size_t>(weights.size(0)+1) &&
                !offsets.empty() && offsets.front()==0 && offsets.back()==input.size(0) &&
                std::is_sorted(offsets.begin(),offsets.end()), "invalid grouped linear row offsets");
    return GroupedFunction::apply(input,weights,offsets);
}
static at::Tensor attention_checked(
    const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
    const at::Tensor& offsets,const at::Tensor& keys,double scale,double dropout_p,bool prepared) {
    payload(q,"q"); payload(k,"k"); payload(v,"v"); matching(q,k); matching(q,v);
    TORCH_CHECK(dropout_p==0.0,"native staged CSR attention does not support dropout");
    TORCH_CHECK(std::isfinite(scale) && std::isfinite(static_cast<float>(scale)),"attention scale must be finite FP32");
    TORCH_CHECK(q.dim()==3 && k.dim()==3 && v.dim()==3 && q.size(1)>0 && q.size(2)>0 &&
        v.size(2)>0 && q.size(1)==k.size(1) && q.size(1)==v.size(1) &&
        q.size(2)==k.size(2) && k.size(0)==v.size(0),"attention requires q[Q,H,D], k[N,H,D], v[N,H,V]");
    TORCH_CHECK(offsets.device()==q.device() && keys.device()==q.device() &&
        offsets.scalar_type()==at::kLong && keys.scalar_type()==at::kLong &&
        offsets.layout()==at::kStrided && keys.layout()==at::kStrided &&
        offsets.dim()==1 && keys.dim()==1 && offsets.numel()==q.size(0)+1,
        "CSR requires CUDA int64 offsets[Q+1] and key_ids[edges] on payload device");
    TORCH_CHECK(!prepared || (offsets.is_contiguous() && keys.is_contiguous()),
        "prepared CSR must have contiguous immutable structure buffers");
    return AttentionFunction::apply(q,k,v,offsets,keys,scale,prepared);
}
at::Tensor csr_scaled_dot_product_attention_cuda(
    const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
    const at::Tensor& offsets,const at::Tensor& keys,double scale,double dropout_p) {
    return attention_checked(q,k,v,offsets,keys,scale,dropout_p,false);
}
at::Tensor csr_scaled_dot_product_attention_cuda_prepared(
    const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
    const at::Tensor& offsets,const at::Tensor& keys,double scale,double dropout_p) {
    return attention_checked(q,k,v,offsets,keys,scale,dropout_p,true);
}
at::Tensor factorized_interaction_cuda(
    const layout::FactorizedInteractionLayout& plan,
    const at::Tensor& x, const at::Tensor& z, const at::Tensor& alpha,
    const at::Tensor& bias, const at::Tensor& gamma,
    at::ScalarType output_dtype, bool direct) {
    layout::validate_factorized_interactions(plan);
    validate_factorized_inputs(plan.view(), x, z, alpha, bias, gamma, output_dtype);
    c10::cuda::CUDAGuard guard(x.device());
    const auto torch_stream = c10::cuda::getCurrentCUDAStream(x.get_device());
    auto native_stream = resources::Stream::borrow(
        torch_stream.stream(), x.get_device());
    auto device_layout = std::make_shared<native::FactorizedInteractionDeviceLayout>(
        plan, x.get_device(), native_stream);
    return factorized_interaction_cuda_prepared(
        device_layout, x, z, alpha, bias, gamma, output_dtype, direct);
}

at::Tensor factorized_interaction_cuda_prepared(
    const std::shared_ptr<native::FactorizedInteractionDeviceLayout>& plan,
    const at::Tensor& x, const at::Tensor& z, const at::Tensor& alpha,
    const at::Tensor& bias, const at::Tensor& gamma,
    at::ScalarType output_dtype, bool direct) {
    TORCH_CHECK(plan != nullptr,
                "factorized interaction requires a prepared device layout");
    validate_factorized_inputs(plan->view(), x, z, alpha, bias, gamma, output_dtype);
    TORCH_CHECK(plan->device() == x.get_device(),
                "prepared factorized interaction and tensors must use one CUDA device");
    return FactorizedInteractionFunction::apply(
        plan, x, z, alpha, bias, gamma,
        static_cast<std::int64_t>(output_dtype), direct ? 1 : 0);
}
} // namespace cellerator::nn::ops
