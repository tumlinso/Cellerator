#include <Cellerator/compiler/profile/represent_mutability_recurrence_and_reuse_evidence_v1.hh>
#include <cmath>
namespace cellerator::compiler::profile::v1 {
namespace {
observed_rate_v1 rate(std::uint64_t hits, std::uint64_t n) noexcept {
    const double p = static_cast<double>(hits) / n;
    const double error = 1.96 * std::sqrt(p * (1.0 - p) / n);
    return {p, p > error ? p - error : 0.0, p + error < 1.0 ? p + error : 1.0};
}
double half_life(double p) noexcept { return p == 0.0 ? 0.0 : std::log(2.0) / p; }
}  // namespace
reuse_profile_status_v1 infer_reuse_profile_evidence_v1(
    const profile_trace_observation_v1 *trace, std::uint64_t count,
    profile_identity_v1 evidence, profile_identity_v1 subject,
    reuse_profile_evidence_v1 *output) noexcept {
    if (trace == nullptr || output == nullptr) return reuse_profile_status_v1::invalid_argument;
    if (count < 2u) return reuse_profile_status_v1::insufficient_trace;
    std::uint64_t sc=0, vc=0, suc=0, oc=0, stable_runs=0, longest=0, executions=0, loops=0;
    for (std::uint64_t i=1; i<count; ++i) {
        const bool s=trace[i].structure_epoch != trace[i-1].structure_epoch;
        const bool v=trace[i].value_generation != trace[i-1].value_generation;
        const bool su=trace[i].support_generation != trace[i-1].support_generation;
        const bool o=trace[i].order_generation != trace[i-1].order_generation;
        sc += s; vc += v; suc += su; oc += o;
        stable_runs = (s || su || o) ? 0u : stable_runs + 1u;
        if (stable_runs > longest) longest = stable_runs;
        executions += trace[i].field_executions;
        loops += trace[i].loop_iterations;
    }
    reuse_profile_evidence_v1 result{}; result.evidence=evidence; result.subject=subject;
    result.observation_count=count; result.transition_count=count-1u;
    result.structure_change=rate(sc,count-1u); result.value_change=rate(vc,count-1u);
    result.support_change=rate(suc,count-1u); result.order_change=rate(oc,count-1u);
    result.structure_mutation_half_life=half_life(result.structure_change.rate);
    result.value_mutation_half_life=half_life(result.value_change.rate);
    result.reuse_horizon=longest; result.recurrence=1.0-result.structure_change.rate;
    result.field_frequency=static_cast<double>(executions)/(count-1u);
    result.mean_loop_count=static_cast<double>(loops)/(count-1u);
    *output=result; return reuse_profile_status_v1::ok;
}
}  // namespace cellerator::compiler::profile::v1
