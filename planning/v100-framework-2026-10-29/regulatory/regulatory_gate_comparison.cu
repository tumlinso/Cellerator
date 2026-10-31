// Temporary V100 comparison driver. It measures two representations of the
// same dense cell/rule eligibility matrix. Build and run only inside a
// root-admitted Device Scheduler job; this file is not part of the default API.
#include <Cellerator/modeling/cuda/regulatory_gates.hh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace m = cellerator::modeling::cuda;
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }

template<class T> struct Buffer {
    T* p = nullptr;
    explicit Buffer(std::size_t n) { if (n) check(cudaMalloc(&p, n * sizeof(T))); }
    ~Buffer() { if (p) cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void upload(const std::vector<T>& v, cudaStream_t s) {
        if (!v.empty()) check(cudaMemcpyAsync(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice, s));
    }
};

template<class F> double resident_ms(F&& submit, cudaStream_t stream,
                                     cudaEvent_t begin, cudaEvent_t end,
                                     unsigned warmup, unsigned repeats) {
    for (unsigned i = 0; i < warmup; ++i) check(submit());
    check(cudaStreamSynchronize(stream));
    check(cudaEventRecord(begin, stream));
    for (unsigned i = 0; i < repeats; ++i) check(submit());
    check(cudaEventRecord(end, stream));
    check(cudaEventSynchronize(end));
    float ms = 0;
    check(cudaEventElapsedTime(&ms, begin, end));
    return ms / repeats;
}

int main(int argc, char** argv) {
    const std::uint64_t cells = argc > 1 ? std::stoull(argv[1]) : 4099;
    const std::uint64_t features = argc > 2 ? std::stoull(argv[2]) : 512;
    const std::uint64_t rule_count = argc > 3 ? std::stoull(argv[3]) : 128;
    const unsigned warmup = argc > 4 ? static_cast<unsigned>(std::stoul(argv[4])) : 10;
    const unsigned repeats = argc > 5 ? static_cast<unsigned>(std::stoul(argv[5])) : 100;
    if (!cells || features < 3 || !rule_count || !repeats)
        throw std::invalid_argument("cells/rules/repeats must be positive and features at least three");

    int device = 0;
    check(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, device));
    const std::uint64_t instance_words = (cells + 31) / 32;
    const std::uint64_t input_words = (features + 63) / 64;
    const std::uint64_t access_words = (rule_count + 63) / 64;

    // Deterministic, shared semantic source state. Its identity order is the
    // same for both packed representations.
    std::vector<std::uint8_t> active(cells * features);
    std::vector<std::uint8_t> accessible(cells * rule_count);
    for (std::uint64_t c = 0; c < cells; ++c) {
        for (std::uint64_t f = 0; f < features; ++f)
            active[c * features + f] = ((c * 17 + f * 11 + 3) % 37) < 17;
        for (std::uint64_t r = 0; r < rule_count; ++r)
            accessible[c * rule_count + r] = ((c * 5 + r * 13 + 1) % 19) < 15;
    }
    std::vector<m::RegulatoryRule> definitions;
    for (std::uint64_t r = 0; r < rule_count; ++r) {
        const auto a = r * 5 % features;
        const auto b = (r * 5 + 17) % features;
        auto forbidden = (r * 9 + 31) % features;
        while (forbidden == a || forbidden == b) forbidden = (forbidden + 1) % features;
        definitions.push_back({10000 + r, {a, b}, {forbidden}});
    }
    const m::PackedRegulatoryRules rules(definitions, features);

    std::vector<std::uint64_t> input, access;
    std::vector<m::RegulatoryOccurrence> occurrences;
    const auto standard_started = std::chrono::steady_clock::now();
    input.assign(cells * input_words, 0);
    access.assign(cells * access_words, 0);
    for (std::uint64_t c = 0; c < cells; ++c) {
        for (std::uint64_t f = 0; f < features; ++f)
            if (active[c * features + f]) input[c * input_words + f / 64] |= std::uint64_t{1} << (f % 64);
        for (std::uint64_t r = 0; r < rule_count; ++r)
            if (accessible[c * rule_count + r]) access[c * access_words + r / 64] |= std::uint64_t{1} << (r % 64);
    }
    occurrences.reserve(cells * rule_count);
    for (std::uint64_t r = 0; r < rule_count; ++r)
        for (std::uint64_t c = 0; c < cells; ++c) occurrences.push_back({c, r});
    const auto standard_ready = std::chrono::steady_clock::now();

    std::vector<std::uint32_t> planes(features * instance_words, 0);
    std::vector<std::uint32_t> plane_access(rule_count * instance_words, 0);
    std::vector<std::uint32_t> validity(instance_words, ~std::uint32_t{0});
    const auto valid_tail = cells % 32;
    if (valid_tail) validity.back() = (std::uint32_t{1} << valid_tail) - 1;
    for (std::uint64_t f = 0; f < features; ++f)
        for (std::uint64_t c = 0; c < cells; ++c)
            if (active[c * features + f]) planes[f * instance_words + c / 32] |= std::uint32_t{1} << (c % 32);
    for (std::uint64_t r = 0; r < rule_count; ++r)
        for (std::uint64_t c = 0; c < cells; ++c)
            if (accessible[c * rule_count + r]) plane_access[r * instance_words + c / 32] |= std::uint32_t{1} << (c % 32);
    std::vector<std::uint64_t> required_offsets{0}, forbidden_offsets{0}, required, forbidden;
    for (const auto& rule : definitions) {
        required.insert(required.end(), rule.required_bits.begin(), rule.required_bits.end());
        forbidden.insert(forbidden.end(), rule.forbidden_bits.begin(), rule.forbidden_bits.end());
        required_offsets.push_back(required.size());
        forbidden_offsets.push_back(forbidden.size());
    }
    std::vector<m::RegulatoryBitplaneWork> plane_work;
    plane_work.reserve(rule_count * instance_words);
    for (std::uint64_t r = 0; r < rule_count; ++r)
        for (std::uint64_t w = 0; w < instance_words; ++w) plane_work.push_back({r, w});
    const auto plane_ready = std::chrono::steady_clock::now();

    const auto occ_count = static_cast<std::uint64_t>(occurrences.size());
    const auto occ_output_words = m::PackedRegulatoryState::words(occ_count);
    const auto plane_count = static_cast<std::uint64_t>(plane_work.size());
    Buffer<std::uint64_t> d_offsets(rules.offsets().size()), d_input(input.size()), d_access(access.size());
    Buffer<m::RegulatoryMaskWord> d_masks(rules.masks().size());
    Buffer<m::RegulatoryOccurrence> d_occ(occurrences.size());
    Buffer<std::uint64_t> d_occ_bits(occ_output_words);
    Buffer<m::RegulatoryGateStatus> d_occ_status(1);
    Buffer<std::uint64_t> d_req_offsets(required_offsets.size()), d_forbid_offsets(forbidden_offsets.size());
    Buffer<std::uint64_t> d_req(required.size()), d_forbid(forbidden.size());
    Buffer<std::uint32_t> d_planes(planes.size()), d_plane_access(plane_access.size()), d_valid(validity.size());
    Buffer<m::RegulatoryBitplaneWork> d_plane_work(plane_work.size());
    Buffer<std::uint32_t> d_plane_output(plane_work.size());
    Buffer<m::RegulatoryBitplaneStatus> d_plane_status(1);
    cudaStream_t stream{};
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const auto upload_started = std::chrono::steady_clock::now();
    d_offsets.upload(rules.offsets(), stream); d_masks.upload(rules.masks(), stream);
    d_input.upload(input, stream); d_access.upload(access, stream); d_occ.upload(occurrences, stream);
    d_req_offsets.upload(required_offsets, stream); d_forbid_offsets.upload(forbidden_offsets, stream);
    d_req.upload(required, stream); d_forbid.upload(forbidden, stream);
    d_planes.upload(planes, stream); d_plane_access.upload(plane_access, stream); d_valid.upload(validity, stream);
    d_plane_work.upload(plane_work, stream);
    check(cudaStreamSynchronize(stream));
    const auto upload_ready = std::chrono::steady_clock::now();

    const m::RegulatoryRuleBankView standard_rules{
        d_offsets.p, d_masks.p, rule_count, rules.masks().size(), features};
    const m::RegulatoryStateView standard_state{
        d_input.p, d_access.p, cells, input_words, access_words};
    const m::RegulatoryBitplaneRules packed_rules{
        d_req_offsets.p, d_forbid_offsets.p, d_req.p, d_forbid.p,
        rule_count, required.size(), forbidden.size(), features};
    const m::RegulatoryBitplaneState packed_state{d_planes.p, d_plane_access.p, d_valid.p, instance_words};
    auto standard_submit = [&] {
        return m::evaluate_regulatory_gates(
            standard_rules, standard_state, d_occ.p, occ_count,
            {d_occ_bits.p, occ_output_words, nullptr, 0, d_occ_status.p, false}, stream);
    };
    auto plane_submit = [&] {
        return m::evaluate_regulatory_bitplanes(
            packed_rules, packed_state, d_plane_work.p, plane_count,
            {d_plane_output.p, plane_count, d_plane_status.p}, stream);
    };
    cudaEvent_t start{}, stop{};
    check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
    const double standard_ms = resident_ms(standard_submit, stream, start, stop, warmup, repeats);
    const double plane_ms = resident_ms(plane_submit, stream, start, stop, warmup, repeats);

    m::RegulatoryGateStatus standard_status{};
    m::RegulatoryBitplaneStatus plane_status{};
    check(cudaMemcpy(&standard_status, d_occ_status.p, sizeof(standard_status), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&plane_status, d_plane_status.p, sizeof(plane_status), cudaMemcpyDeviceToHost));
    if (standard_status.flags || plane_status.flags || standard_status.eligible_count > occ_count || plane_status.written != plane_count)
        throw std::runtime_error("device status failure");
    std::vector<std::uint64_t> standard_output(occ_output_words);
    std::vector<std::uint32_t> plane_output(plane_count);
    check(cudaMemcpy(standard_output.data(), d_occ_bits.p, standard_output.size() * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(plane_output.data(), d_plane_output.p, plane_output.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
    std::uint64_t scalar_eligible = 0;
    for (std::uint64_t r = 0; r < rule_count; ++r) for (std::uint64_t c = 0; c < cells; ++c) {
        bool scalar = accessible[c * rule_count + r];
        for (auto f : definitions[r].required_bits) scalar = scalar && active[c * features + f];
        for (auto f : definitions[r].forbidden_bits) scalar = scalar && !active[c * features + f];
        const auto occurrence = r * cells + c;
        const bool a = (standard_output[occurrence / 64] >> (occurrence % 64)) & 1;
        const bool b = (plane_output[r * instance_words + c / 32] >> (c % 32)) & 1;
        if (a != scalar || b != scalar) throw std::runtime_error("scalar/standard/bitplane disagreement");
        scalar_eligible += scalar;
    }
    if (standard_status.eligible_count != scalar_eligible) throw std::runtime_error("eligible count mismatch");

    const double host_standard_ms = std::chrono::duration<double, std::milli>(standard_ready - standard_started).count();
    const double host_plane_ms = std::chrono::duration<double, std::milli>(plane_ready - standard_ready).count();
    const double h2d_ms = std::chrono::duration<double, std::milli>(upload_ready - upload_started).count();
    const auto standard_input_bytes = input.size() * sizeof(std::uint64_t) + access.size() * sizeof(std::uint64_t);
    const auto plane_input_bytes = planes.size() * sizeof(std::uint32_t) + plane_access.size() * sizeof(std::uint32_t)
        + validity.size() * sizeof(std::uint32_t);
    std::cout << std::fixed << std::setprecision(6)
        << "{\"device\":\"" << prop.name << "\",\"sm\":\"" << prop.major << prop.minor
        << "\",\"cells\":" << cells << ",\"features\":" << features << ",\"rules\":" << rule_count
        << ",\"warmup\":" << warmup << ",\"repeats\":" << repeats
        << ",\"checks\":{\"scalar_oracle\":true,\"standard_eligible\":" << standard_status.eligible_count
        << ",\"bitplane_work_words\":" << plane_status.written << "}"
        << ",\"resident_ms_per_call\":{\"occurrence\":" << standard_ms << ",\"bitplane\":" << plane_ms << "}"
        << ",\"host_prepare_ms\":{\"occurrence_packing_and_worklist\":" << host_standard_ms
        << ",\"bitplane_conversion_and_worklist\":" << host_plane_ms << "}"
        << ",\"h2d_once_ms\":" << h2d_ms
        << ",\"input_bytes\":{\"occurrence_state\":" << standard_input_bytes
        << ",\"bitplane_state\":" << plane_input_bytes << "}"
        << ",\"worklist_bytes\":{\"occurrence\":" << occurrences.size() * sizeof(m::RegulatoryOccurrence)
        << ",\"bitplane\":" << plane_work.size() * sizeof(m::RegulatoryBitplaneWork) << "}"
        << ",\"output_bytes\":{\"occurrence\":" << occ_output_words * sizeof(std::uint64_t)
        << ",\"bitplane\":" << plane_work.size() * sizeof(std::uint32_t) << "}}\n";
    check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
    check(cudaStreamDestroy(stream));
}
