#include "fixture.hh"
int main() {
    nf::support_contract support; support.structure={1,0}; support.epoch={1}; support.activity={{2,0},{1}};
    assert(nf::validate_support(support)==nf::status::success);
    assert(!nf::may_drop_response(nf::contribution_kind::numerical_zero));
    assert(nf::may_drop_response(nf::contribution_kind::predicate_excluded));
    assert(nf::invalidation_for(support,support,true)==(nf::rebind_values|nf::invalidate_primal));
    auto changed=support; changed.activity.generation.value++;
    assert(nf::invalidation_for(support,changed,false)==(nf::refresh_activity|nf::invalidate_primal));
    changed.realization=nf::support_realization::compact_exact_active;
    assert(nf::invalidation_for(support,changed,false)&nf::rebuild_projection);
    changed=support; changed.arithmetic_error_bound=1e-6;
    assert(nf::invalidation_for(support,changed,false)&nf::rebuild_projection);
    changed=support; changed.epoch.value++;
    assert(nf::invalidation_for(support,changed,false)&nf::rebuild_structure);
    changed=support; changed.realization=nf::support_realization::approximate_drop;
    assert(nf::validate_support(changed)==nf::status::invalid_contract);
    changed.accuracy=nf::approximation_kind::empirical;
    assert(nf::validate_support(changed)==nf::status::success);
    changed.preserve_nonfinite_exclusion=false;
    assert(nf::validate_support(changed)==nf::status::invalid_contract);
}
