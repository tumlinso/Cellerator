#include <Cellerator/nn/ops.hh>
#include <Cellerator/modeling/cuda/attention.hh>
#include <Cellerator/modeling/cuda/grouped_linear.hh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/autograd/custom_function.h>
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <thread>
#include <memory>
#include <mutex>

namespace cellerator::nn::ops {
namespace {
namespace native = cellerator::modeling::cuda;
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
native::ConstDeviceView view(const at::Tensor& t) {
    return {t.const_data_ptr(), static_cast<std::size_t>(t.nbytes()), t.get_device()};
}
native::DeviceView writable(const at::Tensor& t) {
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
        std::vector<native::ConstDeviceView> views;
        for (std::int64_t e = 0; e < w.size(0); ++e) views.push_back(view(w[e]));
        state->executor = std::make_unique<native::GroupedLinear>(
            x.size(1), w.size(1), std::move(offsets), std::move(views),
            x.scalar_type() == at::kHalf ? native::DType::float16 : native::DType::float32,
            x.get_device());
        record({input, weights, x, w, y}, stream);
        state->executor->forward(view(x), writable(y), native::Stream::borrow(stream.stream(), x.get_device()));
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
        std::vector<native::DeviceView> views;
        for (std::int64_t e = 0; e < w.size(0); ++e) views.push_back(writable(dw[e]));
        record(saved, stream); record({gradients[0], dy, dx, dw}, stream);
        state->executor->backward(view(x), view(dy), writable(dx), views,
            native::GradientMode::overwrite, native::GradientMode::overwrite,
            native::Stream::borrow(stream.stream(), x.get_device()));
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
        native::check(status, "native staged CSR attention forward");
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
            native::check(status,"native staged CSR attention backward");
        }
        auto restore = [&](const at::Tensor& t) { return t.permute({1,0,2}).to(s[0].scalar_type()).contiguous(); };
        auto gq=restore(dq), gk=restore(dk), gv=restore(dv);
        record({gq,gk,gv},stream);
        return {gq,gk,gv,at::Tensor(),at::Tensor(),at::Tensor(),at::Tensor()};
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
} // namespace cellerator::nn::ops
