#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/modeling/cuda/memory.hh>
#include <Cellerator/modeling/cuda/runtime.hh>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>

using namespace cellerator::modeling::cuda;
template<class T> DeviceBuffer upload(const std::vector<T>& x, const Stream& s) {
    DeviceBuffer b(x.size()*sizeof(T), s.device());
    if (!x.empty()) b.upload_async(x.data(), b.bytes(), s);
    s.synchronize(); return b;
}
template<class T> std::vector<T> download(const DeviceBuffer& b, std::size_t n, const Stream& s) {
    std::vector<T> x(n); if(n) b.download_async(x.data(), b.bytes(), s); s.synchronize(); return x;
}

template<class S> void check_case(std::int64_t channels) {
    constexpr bool is_half=std::is_same<S,__half>::value;
    constexpr std::int64_t rows=3;
    std::vector<S> initial(static_cast<std::size_t>(rows*channels));
    std::vector<float> tau(static_cast<std::size_t>(channels));
    for (std::int64_t c=0;c<channels;++c) tau[static_cast<std::size_t>(c)]=0.5f+0.125f*float(c%9);
    // Update slot 2 with repeated-channel impulses, then slot 0 with none;
    // slot 1 stays lazy. Event times are ordered within the updated slot.
    Stream stream(0);
    auto entities=upload<std::int64_t>({2,0}, stream);
    auto offsets=upload<std::int64_t>({0,3,3},stream);
    auto event_channels=upload<std::int64_t>({0,0,channels-1},stream);
    auto amplitudes=upload<float>({1.25f,-0.125f,0.75f},stream);
    auto event_times=upload<float>({0.5f,1.25f,2.5f},stream);
    auto values=upload(initial,stream);
    auto anchors=upload<float>({0,0,0},stream);
    auto taus=upload(tau,stream);
    EventMemoryView<S> m{static_cast<S*>(values.data()),initial.size(),static_cast<float*>(anchors.data()),3,
        static_cast<const float*>(taus.data()),tau.size(),rows,channels};
    cudaError_t status;
    if constexpr(std::is_same_v<S,__half>) status=apply_events_fp16(m,2,
        static_cast<std::int64_t*>(entities.data()),static_cast<std::int64_t*>(offsets.data()),
        static_cast<std::int64_t*>(event_channels.data()),static_cast<float*>(amplitudes.data()),
        static_cast<float*>(event_times.data()),3,3.f,stream.get());
    else status=apply_events_fp32(m,2,static_cast<std::int64_t*>(entities.data()),
        static_cast<std::int64_t*>(offsets.data()),static_cast<std::int64_t*>(event_channels.data()),
        static_cast<float*>(amplitudes.data()),static_cast<float*>(event_times.data()),3,3.f,stream.get());
    check(status,"apply events"); stream.synchronize();
    auto got=download<S>(values,initial.size(),stream);
    auto anchor=download<float>(anchors,rows,stream);
    assert((anchor==std::vector<float>{3,0,3}));
    for(std::int64_t c=0;c<channels;++c){
        // Independent scalar recurrence, intentionally handling channel 0 ==
        // channels-1 as a single channel when the width is one.
        float expected=0.f, previous=0.f;
        const std::int64_t event_cs[3]={0,0,channels-1};
        const float event_as[3]={1.25f,-0.125f,0.75f};
        const float event_ts[3]={0.5f,1.25f,2.5f};
        for(int j=0;j<3;++j){
            expected*=std::exp(-(event_ts[j]-previous)/tau[static_cast<std::size_t>(c)]);
            if(event_cs[j]==c) expected+=event_as[j];
            previous=event_ts[j];
        }
        expected*=std::exp(-(3.f-previous)/tau[static_cast<std::size_t>(c)]);
        float value;
        if constexpr(std::is_same_v<S,__half>) value=__half2float(got[static_cast<std::size_t>(2*channels+c)]);
        else value=got[static_cast<std::size_t>(2*channels+c)];
        float tol=is_half?2e-3f:2e-5f;
        if(std::abs(value-expected)>tol*std::max(1.f,std::abs(expected)))
            std::cerr<<"channels="<<channels<<" channel="<<c<<" actual="<<value
                     <<" expected="<<expected<<" tolerance="<<tol<<'\n';
        assert(std::abs(value-expected)<=tol*std::max(1.f,std::abs(expected)));
        assert(got[static_cast<std::size_t>(channels+c)]==S{} && got[static_cast<std::size_t>(c)]==S{});
    }
    auto read_entities=upload<std::int64_t>({2,1,2},stream);
    DeviceBuffer output(3*static_cast<std::size_t>(channels)*sizeof(S),0);
    if constexpr(std::is_same_v<S,__half>) check(read_traces_fp16(m,
        static_cast<std::int64_t*>(read_entities.data()),3,4.f,static_cast<__half*>(output.data()),3*channels,stream.get()),"read fp16");
    else check(read_traces_fp32(m,static_cast<std::int64_t*>(read_entities.data()),3,4.f,
        static_cast<float*>(output.data()),3*channels,stream.get()),"read fp32");
    auto observed=download<S>(output,3*static_cast<std::size_t>(channels),stream);
    for(std::int64_t c=0;c<channels;++c){
        std::size_t i=static_cast<std::size_t>(2*channels+c);
        float stored;
        if constexpr(std::is_same_v<S,__half>) stored=__half2float(got[i]); else stored=got[i];
        float expected=stored*std::exp(-1.f/tau[static_cast<std::size_t>(c)]), value;
        if constexpr(std::is_same_v<S,__half>) value=__half2float(observed[static_cast<std::size_t>(c)]);
        else value=observed[static_cast<std::size_t>(c)];
        float tol=is_half?3e-3f:2e-5f;
        assert(std::abs(value-expected)<=tol*std::max(1.f,std::abs(expected)));
        assert(observed[static_cast<std::size_t>(channels+c)]==S{});
        assert(observed[static_cast<std::size_t>(2*channels+c)]==observed[static_cast<std::size_t>(c)]);
    }
    assert(download<float>(anchors,rows,stream)==anchor); // read is lazy and non-mutating
    check(apply_events_fp32({nullptr,0,nullptr,0,static_cast<const float*>(taus.data()),tau.size(),0,channels},
        0,nullptr,nullptr,nullptr,nullptr,nullptr,0,3.f,stream.get()),"empty update");
    if constexpr(is_half) check(read_traces_fp16(m,nullptr,0,4.f,nullptr,0,stream.get()),"empty fp16 read");
    else check(read_traces_fp32(m,nullptr,0,4.f,nullptr,0,stream.get()),"empty fp32 read");
    if constexpr(std::is_same_v<S,__half>) {
        assert(read_traces_fp16(m,nullptr,-1,4.f,nullptr,0,stream.get())==cudaErrorInvalidValue);
        assert(read_traces_fp16(m,static_cast<std::int64_t*>(read_entities.data()),3,4.f,
            static_cast<__half*>(output.data()),1,stream.get())==cudaErrorInvalidValue);
    } else {
        assert(read_traces_fp32(m,nullptr,-1,4.f,nullptr,0,stream.get())==cudaErrorInvalidValue);
        assert(read_traces_fp32(m,static_cast<std::int64_t*>(read_entities.data()),3,4.f,
            static_cast<float*>(output.data()),1,stream.get())==cudaErrorInvalidValue);
    }
}
int main(){
    check_case<float>(1); check_case<float>(33); check_case<float>(257);
    check_case<__half>(1); check_case<__half>(33); check_case<__half>(257);
    std::cout<<"event-memory fp16/fp32 device oracle passed\n";
}
