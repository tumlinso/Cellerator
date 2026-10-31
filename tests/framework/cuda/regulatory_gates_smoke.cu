#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/modeling/cuda/regulatory_gates.hh>
#include <cuda_runtime.h>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
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
    constexpr std::uint64_t tail_cells = 67;
    const auto bitplane_words = (tail_cells + 31) / 32;
    assert(bitplane_words == 3);
    // The ordinary state helper packs one bit per input feature into uint64s;
    // it must not be reused to count independent-cell uint32 bitplane words.
    assert(m::PackedRegulatoryState::words(tail_cells) == 2);
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

void check_device_bitplanes(cudaStream_t stream) {
    // The two device entry points receive equivalent predicates and states.
    // The standard path returns one bit per occurrence; the bitplane path
    // returns 32 independent cell results per (rule, instance-word).
    constexpr std::uint64_t cells = 67;
    constexpr std::uint64_t features = 130;
    constexpr std::uint64_t rules_count = 66;
    // Bitplanes use uint32 per 32 instances; PackedRegulatoryState::words()
    // counts uint64 input bits and therefore divides by 64.
    const auto instance_words = (cells + 31) / 32;
    std::vector<m::RegulatoryRule> rule_defs;
    std::vector<std::vector<bool>> active(cells, std::vector<bool>(features));
    std::vector<std::vector<bool>> accessible(cells, std::vector<bool>(rules_count));
    for (std::uint64_t cell = 0; cell < cells; ++cell) {
        for (std::uint64_t feature = 0; feature < features; ++feature)
            active[cell][feature] = ((cell * 19 + feature * 7 + 3) % 23) < 11;
        for (std::uint64_t rule = 0; rule < rules_count; ++rule)
            accessible[cell][rule] = (cell + rule * 3) % 7 != 0;
    }
    for (std::uint64_t rule = 0; rule < rules_count; ++rule) {
        m::RegulatoryRule spec{5000 + rule, {}, {}};
        spec.required_bits.push_back(rule % features);
        const auto second = (rule * 17 + 64) % features;
        if (second != spec.required_bits.front() && rule % 3 == 0)
            spec.required_bits.push_back(second);
        auto forbidden = (rule * 11 + 37) % features;
        while (std::find(spec.required_bits.begin(), spec.required_bits.end(), forbidden)
               != spec.required_bits.end())
            forbidden = (forbidden + 1) % features;
        if (rule % 4 != 1) spec.forbidden_bits.push_back(forbidden);
        rule_defs.push_back(std::move(spec));
    }
    const m::PackedRegulatoryRules bank(rule_defs, features);

    // Build the occurrence API's instance-major state and scalar reference.
    std::vector<std::uint64_t> input_words, access_words;
    input_words.reserve(cells * m::PackedRegulatoryState::words(features));
    access_words.reserve(cells * m::PackedRegulatoryState::words(rules_count));
    for (std::uint64_t cell = 0; cell < cells; ++cell) {
        m::PackedRegulatoryState packed(features, rules_count, false);
        for (std::uint64_t feature = 0; feature < features; ++feature)
            if (active[cell][feature]) packed.set_input(feature, true);
        for (std::uint64_t rule = 0; rule < rules_count; ++rule)
            if (accessible[cell][rule]) packed.set_accessible(rule, true);
        input_words.insert(input_words.end(), packed.inputs().begin(), packed.inputs().end());
        access_words.insert(access_words.end(), packed.accessibility().begin(), packed.accessibility().end());
    }

    std::vector<m::RegulatoryOccurrence> occurrences;
    occurrences.reserve(cells * rules_count + 2);
    for (std::uint64_t cell = 0; cell < cells; ++cell)
        for (std::uint64_t rule = 0; rule < rules_count; ++rule)
            occurrences.push_back({cell, rule});
    occurrences.push_back(occurrences[0]); // Duplicates remain separate work.
    occurrences.push_back(occurrences[occurrences.size() / 2]);

    std::vector<std::uint64_t> expected;
    expected.reserve(occurrences.size());
    for (const auto item : occurrences) {
        bool eligible = accessible[item.instance][item.rule];
        // Rule vectors are the independent scalar definition, separate from
        // both the standard packed-word bank and feature-plane preparation.
        for (const auto bit : rule_defs[item.rule].required_bits)
            eligible = eligible && active[item.instance][bit];
        for (const auto bit : rule_defs[item.rule].forbidden_bits)
            eligible = eligible && !active[item.instance][bit];
        expected.push_back(eligible ? 1 : 0);
    }

    Buffer<std::uint64_t> d_offsets(bank.offsets().size()), d_inputs(input_words.size()),
        d_access(access_words.size()), d_occurrence_bits(m::PackedRegulatoryState::words(occurrences.size()));
    Buffer<m::RegulatoryMaskWord> d_masks(bank.masks().size());
    Buffer<m::RegulatoryOccurrence> d_occurrences(occurrences.size());
    Buffer<m::RegulatoryGateStatus> d_occurrence_status(1);
    d_offsets.upload(bank.offsets(), stream);
    d_masks.upload(bank.masks(), stream);
    d_inputs.upload(input_words, stream);
    d_access.upload(access_words, stream);
    d_occurrences.upload(occurrences, stream);
    const m::RegulatoryRuleBankView standard_rules{
        d_offsets.data, d_masks.data, rules_count, bank.masks().size(), features};
    const m::RegulatoryStateView standard_state{
        d_inputs.data, d_access.data, cells,
        m::PackedRegulatoryState::words(features), m::PackedRegulatoryState::words(rules_count)};
    check(m::evaluate_regulatory_gates(
        standard_rules, standard_state, d_occurrences.data, occurrences.size(),
        {d_occurrence_bits.data, m::PackedRegulatoryState::words(occurrences.size()), nullptr, 0,
         d_occurrence_status.data, false}, stream));

    // Convert the same input booleans and accessibility to feature-major
    // planes. Tail masks are explicit and all unused tail bits remain clear.
    std::vector<std::uint32_t> feature_planes(features * instance_words, 0);
    std::vector<std::uint32_t> access_planes(rules_count * instance_words, 0);
    std::vector<std::uint32_t> valid_instances(instance_words, ~std::uint32_t{0});
    const auto tail = static_cast<unsigned>(cells % 32);
    if (tail) valid_instances.back() = (std::uint32_t{1} << tail) - 1;
    for (std::uint64_t feature = 0; feature < features; ++feature)
        for (std::uint64_t cell = 0; cell < cells; ++cell)
            if (active[cell][feature])
                feature_planes[feature * instance_words + cell / 32] |= std::uint32_t{1} << (cell % 32);
    for (std::uint64_t rule = 0; rule < rules_count; ++rule)
        for (std::uint64_t cell = 0; cell < cells; ++cell)
            if (accessible[cell][rule])
                access_planes[rule * instance_words + cell / 32] |= std::uint32_t{1} << (cell % 32);

    std::vector<std::uint64_t> required_offsets{0}, forbidden_offsets{0};
    std::vector<std::uint64_t> required_features, forbidden_features;
    for (std::uint64_t rule = 0; rule < rules_count; ++rule) {
        required_features.insert(required_features.end(),
                                 rule_defs[rule].required_bits.begin(), rule_defs[rule].required_bits.end());
        forbidden_features.insert(forbidden_features.end(),
                                  rule_defs[rule].forbidden_bits.begin(), rule_defs[rule].forbidden_bits.end());
        required_offsets.push_back(required_features.size());
        forbidden_offsets.push_back(forbidden_features.size());
    }
    std::vector<m::RegulatoryBitplaneWork> packed_work;
    for (std::uint64_t rule = 0; rule < rules_count; ++rule)
        for (std::uint64_t word = 0; word < instance_words; ++word)
            packed_work.push_back({rule, word});
    packed_work.push_back(packed_work.front());
    Buffer<std::uint32_t> d_features(feature_planes.size()), d_access_planes(access_planes.size()),
        d_valid(valid_instances.size()), d_packed_output(packed_work.size());
    Buffer<std::uint64_t> d_required_offsets(required_offsets.size()),
        d_forbidden_offsets(forbidden_offsets.size()), d_required(required_features.size()),
        d_forbidden(forbidden_features.size());
    Buffer<m::RegulatoryBitplaneWork> d_packed_work(packed_work.size());
    Buffer<m::RegulatoryBitplaneStatus> d_packed_status(1);
    d_features.upload(feature_planes, stream);
    d_access_planes.upload(access_planes, stream);
    d_valid.upload(valid_instances, stream);
    d_required_offsets.upload(required_offsets, stream);
    d_forbidden_offsets.upload(forbidden_offsets, stream);
    d_required.upload(required_features, stream);
    d_forbidden.upload(forbidden_features, stream);
    d_packed_work.upload(packed_work, stream);
    const m::RegulatoryBitplaneRules plane_rules{
        d_required_offsets.data, d_forbidden_offsets.data, d_required.data, d_forbidden.data,
        rules_count, required_features.size(), forbidden_features.size(), features};
    const m::RegulatoryBitplaneState plane_state{
        d_features.data, d_access_planes.data, d_valid.data, instance_words};
    check(m::evaluate_regulatory_bitplanes(
        plane_rules, plane_state, d_packed_work.data, packed_work.size(),
        {d_packed_output.data, packed_work.size(), d_packed_status.data}, stream));
    check(cudaStreamSynchronize(stream));

    m::RegulatoryGateStatus occurrence_status{};
    m::RegulatoryBitplaneStatus packed_status{};
    check(cudaMemcpy(&occurrence_status, d_occurrence_status.data, sizeof(occurrence_status), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&packed_status, d_packed_status.data, sizeof(packed_status), cudaMemcpyDeviceToHost));
    assert(occurrence_status.flags == 0);
    assert(packed_status.flags == 0 && packed_status.written == packed_work.size());
    std::vector<std::uint64_t> occurrence_bits(m::PackedRegulatoryState::words(occurrences.size()));
    std::vector<std::uint32_t> packed_output(packed_work.size());
    check(cudaMemcpy(occurrence_bits.data(), d_occurrence_bits.data,
                     occurrence_bits.size() * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(packed_output.data(), d_packed_output.data,
                     packed_output.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
    for (std::size_t index = 0; index < occurrences.size(); ++index) {
        const bool from_occurrence = (occurrence_bits[index / 64] >> (index % 64)) & 1;
        assert(from_occurrence == static_cast<bool>(expected[index]));
        const auto item = occurrences[index];
        const auto packed_item = item.rule * instance_words + item.instance / 32;
        const bool from_bitplane = (packed_output[packed_item] >> (item.instance % 32)) & 1;
        if (from_bitplane != from_occurrence) {
            std::cerr << "regulatory bitplane mismatch: occurrence=" << index
                      << " cell=" << item.instance << " rule=" << item.rule
                      << " word=" << item.instance / 32 << " lane=" << item.instance % 32
                      << " packed_item=" << packed_item
                      << " occurrence_result=" << from_occurrence
                      << " bitplane_result=" << from_bitplane
                      << " scalar_result=" << expected[index]
                      << " output_word=0x" << std::hex << packed_output[packed_item] << std::dec
                      << " valid_word=0x" << std::hex << valid_instances[item.instance / 32] << std::dec
                      << " accessibility=" << accessible[item.instance][item.rule] << " required=";
            for (auto feature : rule_defs[item.rule].required_bits)
                std::cerr << feature << ':' << active[item.instance][feature] << ',';
            std::cerr << " forbidden=";
            for (auto feature : rule_defs[item.rule].forbidden_bits)
                std::cerr << feature << ':' << active[item.instance][feature] << ',';
            std::cerr << '\n';
            assert(from_bitplane == from_occurrence);
        }
    }
    assert(packed_output.back() == packed_output.front());
    assert((packed_output[(rules_count - 1) * instance_words + instance_words - 1]
            & ~valid_instances.back()) == 0);

    // A null accessibility plane is the documented all-open case.
    const m::RegulatoryBitplaneState all_open_state{
        d_features.data, nullptr, d_valid.data, instance_words};
    check(m::evaluate_regulatory_bitplanes(
        plane_rules, all_open_state, d_packed_work.data, packed_work.size(),
        {d_packed_output.data, packed_work.size(), d_packed_status.data}, stream));
    check(cudaStreamSynchronize(stream));
    check(cudaMemcpy(&packed_status, d_packed_status.data, sizeof(packed_status), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(packed_output.data(), d_packed_output.data,
                     packed_output.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
    assert(packed_status.flags == 0 && packed_status.written == packed_work.size());
    for (std::size_t wi = 0; wi < packed_work.size(); ++wi) {
        const auto item = packed_work[wi];
        for (std::uint64_t lane = 0; lane < 32; ++lane) {
            const auto cell = item.instance_word * 32 + lane;
            const bool valid = (valid_instances[item.instance_word] >> lane) & 1;
            bool reference = valid;
            for (std::uint64_t p = required_offsets[item.rule]; p < required_offsets[item.rule + 1]; ++p)
                reference = reference && active[cell][required_features[p]];
            for (std::uint64_t p = forbidden_offsets[item.rule]; p < forbidden_offsets[item.rule + 1]; ++p)
                reference = reference && !active[cell][forbidden_features[p]];
            assert(static_cast<bool>((packed_output[wi] >> lane) & 1) == reference);
        }
    }

    // Content errors must fail closed; an undersized host extent must fail
    // before launch, and zero work must still publish an empty status.
    auto invalid_features = required_features;
    invalid_features[0] = features;
    d_required.upload(invalid_features, stream);
    check(m::evaluate_regulatory_bitplanes(
        plane_rules, plane_state, d_packed_work.data, packed_work.size(),
        {d_packed_output.data, packed_work.size(), d_packed_status.data}, stream));
    check(cudaStreamSynchronize(stream));
    check(cudaMemcpy(&packed_status, d_packed_status.data, sizeof(packed_status), cudaMemcpyDeviceToHost));
    assert(packed_status.flags == m::regulatory_gate_invalid_rule && packed_status.written == 0);
    d_required.upload(required_features, stream);

    auto invalid_work = packed_work;
    invalid_work[0].instance_word = instance_words;
    d_packed_work.upload(invalid_work, stream);
    check(m::evaluate_regulatory_bitplanes(
        plane_rules, plane_state, d_packed_work.data, invalid_work.size(),
        {d_packed_output.data, invalid_work.size(), d_packed_status.data}, stream));
    check(cudaStreamSynchronize(stream));
    check(cudaMemcpy(&packed_status, d_packed_status.data, sizeof(packed_status), cudaMemcpyDeviceToHost));
    assert(packed_status.flags == m::regulatory_gate_invalid_occurrence && packed_status.written == 0);

    assert(m::evaluate_regulatory_bitplanes(
        plane_rules, plane_state, d_packed_work.data, packed_work.size(),
        {d_packed_output.data, packed_work.size() - 1, d_packed_status.data}, stream)
        == cudaErrorInvalidValue);
    check(m::evaluate_regulatory_bitplanes(
        plane_rules, plane_state, nullptr, 0,
        {nullptr, 0, d_packed_status.data}, stream));
    check(cudaStreamSynchronize(stream));
    check(cudaMemcpy(&packed_status, d_packed_status.data, sizeof(packed_status), cudaMemcpyDeviceToHost));
    assert(packed_status.flags == 0 && packed_status.written == 0);
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
    check_device_bitplanes(stream);
    check(cudaStreamDestroy(stream));
    std::cout << "regulatory CUDA worklist and bitplane gates match independent scalar oracle; bounds, tails and duplicate work passed\n";
}
