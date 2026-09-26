#pragma once
#include <Cellerator/compute/operation/indexed_mechanism/incidence.hh>
namespace cellerator::compute::operation::indexed {
enum class evaluator_status { success, invalid_contract, duplicate_registration,
    missing_evaluator, incompatible_incidence, unsupported_action, allocation_failure };
struct evaluator_registration {
    nf1::compiled_block block{};
    std::uint64_t candidate_id=0;
    const void* provider_state=nullptr;
};
// Cold local catalogue of consumer-supplied compiled blocks. No global backend
// registry or new execution owner. Contract spans/provider state stay borrowed.
class evaluator_catalogue {
public:
    evaluator_status add(const evaluator_registration&) noexcept;
    const std::vector<evaluator_registration>& entries() const noexcept { return entries_; }
private:
    std::vector<evaluator_registration> entries_;
};
struct evaluator_group {
    const argument_incidence* incidence=nullptr;
    const void* provider_state=nullptr;
    std::vector<std::uint64_t> mechanism_indices;
};
class grouped_evaluators {
public:
    grouped_evaluators()=default;
    grouped_evaluators(const grouped_evaluators&)=delete;
    grouped_evaluators& operator=(const grouped_evaluators&)=delete;
    grouped_evaluators(grouped_evaluators&&) noexcept=default;
    grouped_evaluators& operator=(grouped_evaluators&&) noexcept=default;
    // Stable contiguous grouping preserves mechanism order and accumulation
    // order. No arbitrary permutation is justified by matching evaluator IDs.
    evaluator_status prepare(const argument_incidence&,const evaluator_catalogue&,
        nf1::capability action=nf1::forward) noexcept;
    execution::program::prepared_program_v2 program() const noexcept;
    const std::vector<evaluator_group>& groups() const noexcept { return groups_; }
private:
    std::vector<evaluator_group> groups_;
    std::vector<execution::program::prepared_stage_v2> stages_;
    std::vector<std::uint64_t> dependencies_;
};
} // namespace cellerator::compute::operation::indexed
