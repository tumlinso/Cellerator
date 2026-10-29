#include <Cellerator/modeling/cuda/regulatory_gates.hh>
#include <cuda_runtime.h>
#include <cstddef>

namespace cellerator::modeling::cuda {
namespace {
__global__ void evaluate(RegulatoryRuleBankView bank, RegulatoryStateView states,
                          const RegulatoryOccurrence* worklist, std::uint64_t count,
                          RegulatoryGateOutput output) {
    const auto input_words = bank.input_bit_count / 64 + (bank.input_bit_count % 64 != 0);
    for (std::uint64_t i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += std::uint64_t(gridDim.x) * blockDim.x) {
        const auto item = worklist[i];
        if (item.instance >= states.instance_count || item.rule >= bank.rule_count) {
            atomicOr(&output.status->flags, regulatory_gate_invalid_occurrence);
            continue;
        }
        const auto begin = bank.offsets[item.rule], end = bank.offsets[item.rule + 1];
        if (begin > end || end > bank.mask_count) {
            atomicOr(&output.status->flags, regulatory_gate_invalid_rule);
            continue;
        }
        bool eligible = ((states.accessibility[item.instance * states.accessibility_word_stride + item.rule / 64]
                          >> (item.rule % 64)) & 1) != 0;
        for (auto mask_index = begin; mask_index < end; ++mask_index) {
            const auto mask = bank.masks[mask_index];
            const bool bad_tail = bank.input_bit_count % 64 && mask.word == input_words - 1
                && ((mask.required | mask.forbidden) >> (bank.input_bit_count % 64));
            if (mask.word >= input_words || (mask.required & mask.forbidden) || bad_tail) {
                atomicOr(&output.status->flags, regulatory_gate_invalid_rule);
                eligible = false;
                continue;
            }
            const auto active = states.inputs[item.instance * states.input_word_stride + mask.word];
            eligible = eligible && ((active & mask.required) == mask.required)
                                 && ((active & mask.forbidden) == 0);
        }
        if (!eligible) continue;
        atomicOr(reinterpret_cast<unsigned long long*>(output.eligibility_bits + i / 64),
                 1ULL << (i % 64));
        const auto position = atomicAdd(
            reinterpret_cast<unsigned long long*>(&output.status->eligible_count), 1ULL);
        if (output.compact) {
            if (position < output.compact_capacity) output.compact_occurrence_indices[position] = i;
            else atomicOr(&output.status->flags, regulatory_gate_insufficient_capacity);
        }
    }
}
__global__ void finish(RegulatoryGateOutput output) {
    if (!threadIdx.x && !blockIdx.x)
        output.status->compact_written = output.compact && output.status->flags == 0
            ? output.status->eligible_count : 0;
}
bool extent_ok(std::uint64_t rows, std::uint64_t stride, std::size_t item_bytes) {
    const auto limit = std::numeric_limits<std::size_t>::max() / item_bytes;
    return !rows || stride <= limit / rows;
}
} // namespace

cudaError_t evaluate_regulatory_gates(RegulatoryRuleBankView bank,
                                      RegulatoryStateView states,
                                      const RegulatoryOccurrence* worklist,
                                      std::uint64_t occurrence_count,
                                      RegulatoryGateOutput output,
                                      cudaStream_t stream) noexcept {
    const auto words = PackedRegulatoryState::words(occurrence_count);
    const auto input_words = PackedRegulatoryState::words(bank.input_bit_count);
    const auto rule_words = PackedRegulatoryState::words(bank.rule_count);
    const auto limit = std::numeric_limits<std::size_t>::max();
    if (!output.status || (occurrence_count && (!worklist || !output.eligibility_bits))
        || output.eligibility_word_capacity < words
        || (bank.rule_count && !bank.offsets) || (bank.mask_count && !bank.masks)
        || (states.instance_count && input_words && !states.inputs)
        || (states.instance_count && rule_words && !states.accessibility)
        || states.input_word_stride < input_words || states.accessibility_word_stride < rule_words
        || (output.compact && output.compact_capacity && !output.compact_occurrence_indices)
        || bank.rule_count >= limit / sizeof(std::uint64_t)
        || bank.mask_count > limit / sizeof(RegulatoryMaskWord)
        || occurrence_count > limit / sizeof(RegulatoryOccurrence)
        || output.compact_capacity > limit / sizeof(std::uint64_t)
        || !extent_ok(states.instance_count, states.input_word_stride, sizeof(std::uint64_t))
        || !extent_ok(states.instance_count, states.accessibility_word_stride, sizeof(std::uint64_t)))
        return cudaErrorInvalidValue;
    auto status = cudaMemsetAsync(output.status, 0, sizeof(RegulatoryGateStatus), stream);
    if (status != cudaSuccess) return status;
    if (words) {
        status = cudaMemsetAsync(output.eligibility_bits, 0, words * sizeof(std::uint64_t), stream);
        if (status != cudaSuccess) return status;
    }
    if (occurrence_count) {
        const auto blocks = static_cast<unsigned>(std::min<std::uint64_t>((occurrence_count + 127) / 128, 65535));
        evaluate<<<blocks,128,0,stream>>>(bank, states, worklist, occurrence_count, output);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
    }
    finish<<<1,1,0,stream>>>(output);
    return cudaGetLastError();
}

namespace {
__global__ void bitplanes(RegulatoryBitplaneRules rules, RegulatoryBitplaneState state,
                           const RegulatoryBitplaneWork* work, std::uint64_t count,
                           RegulatoryBitplaneOutput output) {
    for (std::uint64_t i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
         i<count; i+=std::uint64_t(gridDim.x)*blockDim.x) {
        const auto item=work[i];
        output.eligibility_words[i]=0;
        if (item.rule>=rules.rule_count || item.instance_word>=state.instance_word_count) {
            atomicOr(&output.status->flags,regulatory_gate_invalid_occurrence); continue;
        }
        const auto rb=rules.required_offsets[item.rule], re=rules.required_offsets[item.rule+1];
        const auto fb=rules.forbidden_offsets[item.rule], fe=rules.forbidden_offsets[item.rule+1];
        if (rb>re || re>rules.required_count || fb>fe || fe>rules.forbidden_count) {
            atomicOr(&output.status->flags,regulatory_gate_invalid_rule); continue;
        }
        auto result=state.valid_instances[item.instance_word];
        if (state.accessibility) result &= state.accessibility[item.rule*state.instance_word_count+item.instance_word];
        for (auto p=rb; p<re; ++p) {
            const auto feature=rules.required_features[p];
            if (feature>=rules.feature_count) {
                atomicOr(&output.status->flags,regulatory_gate_invalid_rule); result=0;
            } else result &= state.features[feature*state.instance_word_count+item.instance_word];
        }
        for (auto p=fb; p<fe; ++p) {
            const auto feature=rules.forbidden_features[p];
            if (feature>=rules.feature_count) {
                atomicOr(&output.status->flags,regulatory_gate_invalid_rule); result=0;
            } else result &= ~state.features[feature*state.instance_word_count+item.instance_word];
        }
        output.eligibility_words[i]=result;
    }
}
__global__ void finish_bitplanes(RegulatoryBitplaneStatus* status,std::uint64_t count) {
    if (!threadIdx.x && !blockIdx.x) status->written=status->flags==0 ? count : 0;
}
}

cudaError_t evaluate_regulatory_bitplanes(RegulatoryBitplaneRules rules,
                                          RegulatoryBitplaneState state,
                                          const RegulatoryBitplaneWork* work,
                                          std::uint64_t work_count,
                                          RegulatoryBitplaneOutput output,
                                          cudaStream_t stream) noexcept {
    const auto limit=std::numeric_limits<std::size_t>::max();
    if (!output.status || output.capacity<work_count
        || (work_count && (!work || !output.eligibility_words))
        || (rules.rule_count && (!rules.required_offsets || !rules.forbidden_offsets))
        || (rules.required_count && !rules.required_features)
        || (rules.forbidden_count && !rules.forbidden_features)
        || (state.instance_word_count && !state.valid_instances)
        || (state.instance_word_count && rules.feature_count && !state.features)
        || rules.rule_count>=limit/sizeof(std::uint64_t)
        || rules.required_count>limit/sizeof(std::uint64_t)
        || rules.forbidden_count>limit/sizeof(std::uint64_t)
        || state.instance_word_count>limit/sizeof(std::uint32_t)
        || work_count>limit/sizeof(RegulatoryBitplaneWork)
        || !extent_ok(rules.feature_count,state.instance_word_count,sizeof(std::uint32_t))
        || (state.accessibility && !extent_ok(rules.rule_count,state.instance_word_count,sizeof(std::uint32_t))))
        return cudaErrorInvalidValue;
    auto error=cudaMemsetAsync(output.status,0,sizeof(RegulatoryBitplaneStatus),stream);
    if (error!=cudaSuccess) return error;
    if (work_count) {
        const auto blocks=static_cast<unsigned>(std::min<std::uint64_t>((work_count+127)/128,65535));
        bitplanes<<<blocks,128,0,stream>>>(rules,state,work,work_count,output);
        error=cudaGetLastError(); if (error!=cudaSuccess) return error;
    }
    finish_bitplanes<<<1,1,0,stream>>>(output.status,work_count);
    return cudaGetLastError();
}
} // namespace cellerator::modeling::cuda
