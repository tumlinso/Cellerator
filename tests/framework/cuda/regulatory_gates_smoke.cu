#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/modeling/cuda/regulatory_gates.hh>
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
#include <string>

namespace m = cellerator::modeling::cuda;
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
template<class T> struct Buffer {
    T* data = nullptr;
    explicit Buffer(std::size_t n) { if (n) check(cudaMalloc(&data, n*sizeof(T))); }
    ~Buffer() { if (data) cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void upload(const std::vector<T>& values, cudaStream_t stream) {
        if (!values.empty()) check(cudaMemcpyAsync(data,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice,stream));
    }
};
template<class E, class F> void rejects(F f) {
    bool caught = false;
    try { f(); } catch (const E&) { caught = true; }
    assert(caught);
}

void check_bitplane_host_oracle() {
    // Exhaust all three-feature truth assignments across 32 independent cells,
    // then a second word with only nine valid cells. No cross-cell integer carry.
    const std::vector<std::uint32_t> planes{
        0xaaaaaaaaU,0xaaaaaaaaU, 0xccccccccU,0xccccccccU, 0xf0f0f0f0U,0xf0f0f0f0U};
    const std::uint32_t valid[2]={0xffffffffU,0x1ffU};
    const std::uint32_t accessible[2]={0xfffffff7U,0xffffffffU};
    for (std::size_t word=0;word<2;++word) {
        const auto packed=valid[word]&accessible[word]&planes[word]&planes[4+word]&~planes[2+word];
        std::uint32_t scalar=0;
        for (unsigned cell=0;cell<32;++cell) {
            const bool expected=((valid[word]>>cell)&1) && ((accessible[word]>>cell)&1)
                && ((planes[word]>>cell)&1) && ((planes[4+word]>>cell)&1)
                && !((planes[2+word]>>cell)&1);
            if (expected) scalar|=std::uint32_t{1}<<cell;
        }
        assert(packed==scalar && !(packed&~valid[word]));
        // A second packed transition consumes the first directly.
        assert((packed & planes[word])==packed);
    }
}

int main(int argc, char** argv) {
    check_bitplane_host_oracle();
    std::vector<m::RegulatoryRule> rules;
    for (std::uint64_t i=0; i<66; ++i) rules.push_back({1000+i,{},{}});
    rules[0].required_bits = {0,63,64,129}; rules[0].forbidden_bits = {65};
    rules[1].forbidden_bits = {0,64};
    rules[64].required_bits = {64};
    rules[65].required_bits = {129}; rules[65].forbidden_bits = {128};
    const m::PackedRegulatoryRules bank(rules,130,7);
    assert(bank.rule_ordinal(1065)==65 && bank.structure_epoch()==7);
    rejects<std::invalid_argument>([]{ m::PackedRegulatoryRules({{1,{0},{0}}},1); });
    rejects<std::invalid_argument>([]{ m::PackedRegulatoryRules({{1,{},{}},{1,{},{}}},0); });
    rejects<std::out_of_range>([]{ m::PackedRegulatoryRules({{1,{64},{}}},64); });

    std::vector<std::vector<bool>> active(3,std::vector<bool>(130,false));
    std::vector<std::vector<bool>> open(3,std::vector<bool>(66,true));
    for (auto bit : {0,63,64,129}) active[0][bit]=true;
    active[1][65]=true; active[1][128]=true;
    active[2]=active[0]; active[2][65]=true;
    open[1][2]=false; open[2][65]=false;
    std::vector<m::PackedRegulatoryState> state;
    std::vector<std::uint64_t> input_words, accessibility;
    for (std::size_t instance=0; instance<3; ++instance) {
        state.emplace_back(130,66);
        for (std::size_t bit=0; bit<130; ++bit) state.back().set_input(bit,active[instance][bit]);
        for (std::size_t rule=0; rule<66; ++rule) state.back().set_accessible(rule,open[instance][rule]);
        input_words.insert(input_words.end(),state.back().inputs().begin(),state.back().inputs().end());
        accessibility.insert(accessibility.end(),state.back().accessibility().begin(),state.back().accessibility().end());
    }
    auto inherited = state[0]; inherited.set_accessible(65,false);
    assert(state[0].accessibility()[1] != inherited.accessibility()[1]);
    std::vector<m::RegulatoryOccurrence> work{{0,0},{1,0},{2,0},{0,1},{1,1},{1,2},
        {0,65},{2,65},{0,64},{2,64},{0,0}};
    while (work.size()<70) work.push_back({0,static_cast<std::uint64_t>(work.size()%66)});
    std::vector<std::uint64_t> expected_indices, expected_words(m::PackedRegulatoryState::words(work.size()),0);
    for (std::size_t i=0; i<work.size(); ++i) {
        const auto item=work[i];
        // Independent scalar boolean oracle over original rule definitions.
        bool expected=open[item.instance][item.rule];
        for (auto bit:rules[item.rule].required_bits) expected=expected && active[item.instance][bit];
        for (auto bit:rules[item.rule].forbidden_bits) expected=expected && !active[item.instance][bit];
        bool packed=((accessibility[item.instance*2+item.rule/64]>>(item.rule%64))&1)!=0;
        for (auto p=bank.offsets()[item.rule]; p<bank.offsets()[item.rule+1]; ++p) {
            const auto mask=bank.masks()[p];
            const auto value=input_words[item.instance*3+mask.word];
            packed=packed && (value&mask.required)==mask.required && !(value&mask.forbidden);
        }
        assert(packed==expected);
        if (expected) { expected_words[i/64]|=std::uint64_t{1}<<(i%64); expected_indices.push_back(i); }
    }
    assert(!expected_indices.empty());
    if (argc>1 && std::string(argv[1])=="--host-only") {
        std::cout << "regulatory gate host packing/oracle passed: 70 occurrences, cross-word required/forbidden/accessibility, inherited isolation; bitplane scalar truth table/tail passed\n";
        return 0;
    }
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    Buffer<std::uint64_t> d_offsets(bank.offsets().size()),d_inputs(input_words.size()),d_open(accessibility.size());
    Buffer<m::RegulatoryMaskWord> d_masks(bank.masks().size());
    Buffer<m::RegulatoryOccurrence> d_work(work.size());
    Buffer<std::uint64_t> d_bits(expected_words.size()+1),d_compact(work.size()+1);
    Buffer<m::RegulatoryGateStatus> d_status(1);
    d_offsets.upload(bank.offsets(),stream); d_masks.upload(bank.masks(),stream);
    d_inputs.upload(input_words,stream); d_open.upload(accessibility,stream); d_work.upload(work,stream);
    m::RegulatoryRuleBankView device_bank{d_offsets.data,d_masks.data,66,bank.masks().size(),130};
    m::RegulatoryStateView device_state{d_inputs.data,d_open.data,3,3,2};
    const std::vector<std::uint64_t> bit_sentinel(expected_words.size()+1,~0ULL), compact_sentinel(work.size()+1,999);
    auto run=[&](std::uint64_t count, std::uint64_t capacity, bool compact=true) {
        d_bits.upload(bit_sentinel,stream); d_compact.upload(compact_sentinel,stream);
        check(m::evaluate_regulatory_gates(device_bank,device_state,d_work.data,count,
            {d_bits.data,expected_words.size(),d_compact.data,capacity,d_status.data,compact},stream));
        m::RegulatoryGateStatus status;
        check(cudaMemcpyAsync(&status,d_status.data,sizeof(status),cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream));
        return status;
    };
    auto status=run(work.size(),work.size());
    assert(status.flags==0 && status.eligible_count==expected_indices.size() && status.compact_written==expected_indices.size());
    std::vector<std::uint64_t> bits(expected_words.size()+1), compact(work.size()+1);
    check(cudaMemcpy(bits.data(),d_bits.data,bits.size()*sizeof(std::uint64_t),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(compact.data(),d_compact.data,compact.size()*sizeof(std::uint64_t),cudaMemcpyDeviceToHost));
    assert(bits.back()==~0ULL && compact.back()==999);
    assert(std::equal(expected_words.begin(),expected_words.end(),bits.begin()));
    std::sort(compact.begin(),compact.begin()+status.compact_written);
    assert(std::equal(expected_indices.begin(),expected_indices.end(),compact.begin()));
    status=run(work.size(),1);
    assert(status.flags==m::regulatory_gate_insufficient_capacity && status.compact_written==0);
    assert(status.eligible_count==expected_indices.size());
    status=run(work.size(),0,false);
    assert(status.flags==0 && status.compact_written==0 && status.eligible_count==expected_indices.size());
    assert(run(0,0).eligible_count==0);
    work[0].instance=99; d_work.upload(work,stream);
    status=run(work.size(),work.size());
    assert(status.flags==m::regulatory_gate_invalid_occurrence && status.compact_written==0);
    work[0].instance=0; d_work.upload(work,stream);
    auto bad_masks=bank.masks(); bad_masks[0].word=3; d_masks.upload(bad_masks,stream);
    assert(run(work.size(),work.size()).flags==m::regulatory_gate_invalid_rule);
    assert(m::evaluate_regulatory_gates(device_bank,device_state,d_work.data,work.size(),
        {d_bits.data,0,d_compact.data,work.size(),d_status.data,true},stream)==cudaErrorInvalidValue);
    check(cudaStreamDestroy(stream));
    std::cout << "regulatory CUDA worklist eligibility, compaction, and bounds passed\n";
}
