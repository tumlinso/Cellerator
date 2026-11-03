#pragma once
#include "backends/consumer_fixture.hh"
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace cgs {
inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(e));
}
template<class T> struct DeviceBuffer {
    T* ptr = nullptr;
    std::size_t size = 0;
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t n):size(n) {
        if (n) cuda_check(cudaMalloc(reinterpret_cast<void**>(&ptr),n*sizeof(T)),"cudaMalloc");
    }
    explicit DeviceBuffer(const std::vector<T>& v):DeviceBuffer(v.size()) {
        if (size) cuda_check(cudaMemcpy(ptr,v.data(),size*sizeof(T),cudaMemcpyHostToDevice),"upload");
    }
    DeviceBuffer(const std::vector<T>& v,bool enabled):DeviceBuffer(enabled?v.size():0) {
        if (size) cuda_check(cudaMemcpy(ptr,v.data(),size*sizeof(T),cudaMemcpyHostToDevice),"upload");
    }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&)=delete;
    DeviceBuffer& operator=(const DeviceBuffer&)=delete;
    void clear(cudaStream_t stream) { if(size) cuda_check(cudaMemsetAsync(ptr,0,size*sizeof(T),stream),"clear"); }
    std::vector<T> download() const { std::vector<T> v(size); if(size) cuda_check(cudaMemcpy(v.data(),ptr,size*sizeof(T),cudaMemcpyDeviceToHost),"download"); return v; }
};
struct DeviceFixture {
    using Group=cellerator::ops::mechanism::GatedProductGroup;
    using Output=cellerator::ops::mechanism::OutputIncidence;
    DeviceBuffer<Group> groups;
    DeviceBuffer<std::uint64_t> args,weights_offsets,group_weights_offsets,ng_offsets,ngroups,nf_offsets,nfeatures;
    DeviceBuffer<Output> outputs;
    DeviceBuffer<std::uint16_t> x16,w16;
    DeviceBuffer<float> x32,w32,theta,probs,dy,y,dx,dtheta,dq,dweights;
    DeviceBuffer<std::uint8_t> admitted;
    DeviceBuffer<std::uint32_t> primal,derivative,feature_masks,local_args,status;
    ProblemView p{}; GateView g{}; ResultView result{};
    explicit DeviceFixture(const ConsumerHostFixture& h):
      groups(h.groups),args(h.arguments),weights_offsets(h.weight_offsets),group_weights_offsets(h.group_weight_offsets),
      ng_offsets(h.neighborhood_group_offsets),ngroups(h.neighborhood_groups),nf_offsets(h.neighborhood_feature_offsets),nfeatures(h.neighborhood_features),
      outputs(h.outputs),x16(h.x16,h.dtype==DType::fp16),w16(h.weights16,h.body==BodyKind::grouped_linear&&h.dtype==DType::fp16),
      x32(h.x32,h.dtype==DType::fp32),w32(h.weights32,h.body==BodyKind::grouped_linear&&h.dtype==DType::fp32),
      theta(h.theta,h.body==BodyKind::ordered_product),probs(h.probabilities,h.profile==GateProfile::ste),dy(h.dy),
      y(h.batch*h.coordinates),dx(h.batch*h.coordinates),dtheta(h.body==BodyKind::ordered_product?h.parameters:0),dq(h.batch*h.conditions),dweights(h.body==BodyKind::grouped_linear?h.weights32.size():0),
      admitted(h.admitted),primal(h.primal),derivative(h.derivative),feature_masks(h.feature_group_masks),local_args(h.argument_local_indices),status(1) {
        p=h.problem(); g=h.gates();
        p.topology.groups=groups.ptr; p.topology.arguments=args.ptr; p.topology.outputs=outputs.ptr;
        p.x=h.dtype==DType::fp16?static_cast<void*>(x16.ptr):static_cast<void*>(x32.ptr);
        p.weights=h.dtype==DType::fp16?static_cast<void*>(w16.ptr):static_cast<void*>(w32.ptr);
        p.theta=theta.ptr; p.weight_offsets=weights_offsets.ptr; p.group_weight_offsets=group_weights_offsets.ptr;
        auto& n=p.neighborhoods; n.group_offsets=ng_offsets.ptr; n.groups=ngroups.ptr;
        n.feature_offsets=nf_offsets.ptr; n.features=nfeatures.ptr; n.feature_group_masks=feature_masks.ptr; n.argument_local_indices=local_args.ptr;
        g.probabilities=probs.ptr; g.admitted=admitted.ptr; g.primal_words=primal.ptr; g.derivative_words=derivative.ptr;
        result.y=y.ptr; result.dx=dx.ptr; result.dtheta=dtheta.ptr; result.dq=dq.ptr; result.dweights=dweights.ptr;
    }
    void clear(cudaStream_t stream) { y.clear(stream); dx.clear(stream); dtheta.clear(stream); dq.clear(stream); dweights.clear(stream); status.clear(stream); }
};
}
