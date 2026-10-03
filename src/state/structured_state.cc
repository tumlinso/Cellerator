#include <Cellerator/state/structured_state.hh>
namespace cellerator::state {
bool same_coordinate(const coordinate& a,const coordinate& b) noexcept {
    return ex::same_identity(a.actor,b.actor) && ex::same_identity(a.private_domain,b.private_domain)
        && a.local_slot==b.local_slot && a.incarnation==b.incarnation;
}
bool same_generations(const generations& a,const generations& b) noexcept {
    return ex::same_identity(a.structure,b.structure) && a.epoch.value==b.epoch.value
        && a.values.value==b.values.value && a.activity.value==b.activity.value
        && a.parameters.value==b.parameters.value;
}
bool same_universe(const support_universe& a,const support_universe& b) noexcept {
    return ex::same_identity(a.identity,b.identity) && ex::same_identity(a.structure,b.structure)
        && a.epoch.value==b.epoch.value && a.kind==b.kind && a.extent==b.extent;
}
status validate_support(const support_view& v) noexcept {
    const auto& u=v.universe;
    if(!ex::valid_identity(u.identity) || !ex::valid_identity(u.structure) || !u.epoch.value)
        return status::invalid_identity;
    switch(u.kind) {
    case support_kind::measurement_detection: case support_kind::primal_dependency:
    case support_kind::derivative_dependency: case support_kind::output_contribution:
    case support_kind::source_interval: case support_kind::candidate_nomination: break;
    default: return status::invalid_support;
    }
    if(v.indices.size() && !v.indices.data()) return status::invalid_support;
    for(std::size_t i=0;i<v.indices.size();++i)
        if(v.indices[i]>=u.extent || (i && v.indices[i]<=v.indices[i-1])) return status::invalid_support;
    return status::success;
}
status validate_layout(const structured_layout& layout,std::span<const coordinate> coords) noexcept {
    if(ex::validate_persistent_axis_identity(layout.actors)!=ex::biological_validation_code::ok)
        return status::invalid_identity;
    if(layout.kind!=state_kind::scalar_patch && layout.kind!=state_kind::actor_private)
        return status::unsupported_layout;
    if(layout.row_offsets.size()!=layout.rows.size()+1 || !layout.row_offsets.data()
        || (layout.rows.size() && !layout.rows.data()) || (coords.size() && !coords.data())
        || layout.row_offsets[0]!=0 || layout.row_offsets.back()!=coords.size()) return status::invalid_layout;
    for(std::size_t i=0;i<layout.rows.size();++i) {
        const auto& row=layout.rows[i];
        if(!ex::valid_identity(row.actor)) return status::invalid_identity;
        for(std::size_t j=0;j<i;++j) if(ex::same_identity(row.actor,layout.rows[j].actor)) return status::invalid_identity;
        const auto start=layout.row_offsets[i],end=layout.row_offsets[i+1];
        if(end<start || end>coords.size()) return status::invalid_layout;
        if(layout.kind==state_kind::scalar_patch) {
            if(end-start!=1 || ex::valid_identity(row.private_domain)) return status::invalid_layout;
        } else if(!ex::valid_identity(row.private_domain)) return status::invalid_identity;
        for(auto j=start;j<end;++j) {
            const auto& c=coords[j];
            if(!ex::same_identity(c.actor,row.actor) || !ex::same_identity(c.private_domain,row.private_domain)
                || c.local_slot!=j-start || !c.incarnation) return status::stale_incarnation;
        }
    }
    return status::success;
}
status uniform_shape(const structured_layout& layout,std::uint64_t& rows,std::uint64_t& columns) noexcept {
    if(layout.kind!=state_kind::scalar_patch && layout.kind!=state_kind::actor_private)
        return status::unsupported_layout;
    if(ex::validate_persistent_axis_identity(layout.actors)!=ex::biological_validation_code::ok)
        return status::invalid_identity;
    if(layout.row_offsets.size()!=layout.rows.size()+1 || !layout.row_offsets.data()
        || (layout.rows.size() && !layout.rows.data()) || layout.row_offsets[0]!=0) return status::invalid_layout;
    const auto width=layout.rows.empty()?0:layout.row_offsets[1];
    for(std::size_t i=0;i<layout.rows.size();++i) {
        if(!ex::valid_identity(layout.rows[i].actor)) return status::invalid_identity;
        for(std::size_t j=0;j<i;++j) if(ex::same_identity(layout.rows[i].actor,layout.rows[j].actor))
            return status::invalid_identity;
        if(layout.row_offsets[i+1]<layout.row_offsets[i]
            || layout.row_offsets[i+1]-layout.row_offsets[i]!=width) return status::unsupported_layout;
        if(layout.kind==state_kind::scalar_patch && (width!=1 || ex::valid_identity(layout.rows[i].private_domain)))
            return status::invalid_layout;
        if(layout.kind==state_kind::actor_private && !ex::valid_identity(layout.rows[i].private_domain))
            return status::invalid_identity;
    }
    rows=layout.rows.size(); columns=width;
    return status::success;
}
} // namespace cellerator::state
