#include <Cellerator/packing/strategies/placement.hh>
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
namespace cellerator::packing::strategies {
namespace {
using v2id=compute::operation::v2::stable_id;
bool same(v2id a,v2id b) { return compute::operation::v2::same_stable_id(a,b); }
status endpoints(const problem& p,std::vector<std::uint64_t>& source,std::vector<std::uint64_t>& destination) {
    for(const auto& job:p.work) {
        if(!same(job.evaluator,p.relation_evaluator) || job.arguments.size()!=1 || job.arguments[0].axis!=0)
            return status::unsupported_semantics;
        for(const auto& output:job.outputs) {
            if(output.axis!=0) return status::unsupported_semantics;
            source.push_back(job.arguments[0].index);destination.push_back(output.index);
        }
    }
    return status::success;
}
status publish_orders(const problem& p,physical_orders orders,realization& r) {
    r.source_physical_order=orders.source;r.destination_physical_order=orders.destination;
    return validate(p,r);
}
bool same_job(const ix::mechanism_incidence& a,const ix::mechanism_incidence& b) {
    if(!same(a.instance,b.instance)||!same(a.evaluator,b.evaluator)||a.arguments.size()!=b.arguments.size()
       ||a.outputs.size()!=b.outputs.size()) return false;
    for(std::size_t i=0;i<a.arguments.size();++i) {
        const auto& x=a.arguments[i];const auto& y=b.arguments[i];
        if(x.slot!=y.slot||!same(x.role,y.role)||x.axis!=y.axis||x.index!=y.index) return false;
    }
    for(std::size_t i=0;i<a.outputs.size();++i) {
        const auto& x=a.outputs[i];const auto& y=b.outputs[i];const auto& e=x.effect;const auto& f=y.effect;
        if(x.slot!=y.slot||!same(x.role,y.role)||x.axis!=y.axis||x.index!=y.index||!same(x.assembly_owner,y.assembly_owner)
           ||e.update!=f.update||e.requires_initialized_destination!=f.requires_initialized_destination
           ||e.input_output_aliasing_legal!=f.input_output_aliasing_legal||e.reserved!=f.reserved
           ||e.input_scale_binding_id!=f.input_scale_binding_id||e.destination_scale_binding_id!=f.destination_scale_binding_id) return false;
    }
    return true;
}
void preserve(std::vector<std::uint64_t>& proposed,const std::vector<std::uint64_t>& previous,
              const std::set<std::uint64_t>& touched) {
    std::vector<std::uint64_t> changed;
    for(auto row:proposed) if(touched.contains(row)) changed.push_back(row);
    proposed=previous;std::size_t cursor=0;
    for(auto& row:proposed) if(touched.contains(row)) row=changed[cursor++];
}
}
status two_sided::operator()(const problem& p,realization& out) const {
    realization next;auto s=identity_strategy{}(p,next);if(s!=status::success)return s;
    std::vector<std::uint64_t> src,dst;s=endpoints(p,src,dst);if(s!=status::success)return s;
    std::vector<std::uint64_t> degree(next.destination_order.size()),load(next.source_order.size()),rank(degree.size());
    for(auto row:dst)++degree[row];
    for(auto row:src)++load[row];
    std::stable_sort(next.destination_order.begin(),next.destination_order.end(),[&](auto a,auto b){return degree[a]>degree[b];});
    for(std::size_t i=0;i<rank.size();++i)rank[next.destination_order[i]]=i;
    std::vector<long double> position(load.size());
    for(std::size_t i=0;i<src.size();++i)position[src[i]]+=rank[dst[i]];
    std::stable_sort(next.source_order.begin(),next.source_order.end(),[&](auto a,auto b){
        if(bool(load[a])!=bool(load[b]))return load[a]!=0;
        if(!load[a])return false;
        auto pa=position[a]/load[a],pb=position[b]/load[b];return pa!=pb?pa<pb:load[a]>load[b];});
    s=publish_orders(p,orders,next);if(s==status::success)out=std::move(next);return s;
}
status shared_load_cohorts::operator()(const problem& p,realization& out) const {
    realization next;auto s=identity_strategy{}(p,next);if(s!=status::success)return s;
    if(keys.size()!=p.work.size())return status::capacity;
    std::vector<std::uint64_t> src,dst;s=endpoints(p,src,dst);if(s!=status::success)return s;
    for(std::size_t i=0;i<keys.size();++i)
        if(!same(keys[i].opcode,p.work[i].evaluator))return status::unsupported_semantics;
    auto key=[&](auto i){const auto& k=keys[i];return std::tuple{k.opcode.high,k.opcode.low,k.parameters.high,k.parameters.low,
                                               p.work[i].arguments.size(),p.work[i].arguments[0].index};};
    std::stable_sort(next.operation_order.begin(),next.operation_order.end(),[&](auto a,auto b){return key(a)<key(b);});
    std::vector<bool> used(next.source_order.size());next.source_order.clear();
    for(auto i:next.operation_order) {auto row=p.work[i].arguments[0].index;if(!used[row]){used[row]=true;next.source_order.push_back(row);}}
    for(std::size_t i=0;i<used.size();++i)if(!used[i])next.source_order.push_back(i);
    s=publish_orders(p,orders,next);if(s==status::success)out=std::move(next);return s;
}
status regions(const problem& p,const realization& r,std::uint32_t width,double threshold,std::vector<region>& out) noexcept {
    try {
        auto s=validate(p,r);if(s!=status::success)return s;
        if(!width||!std::isfinite(threshold)||threshold<0||threshold>1)return status::invalid_strategy;
        std::vector<std::uint64_t> src,dst;s=endpoints(p,src,dst);if(s!=status::success)return s;
        std::vector<std::uint64_t> si(r.source_order.size()),di(r.destination_order.size());
        for(std::size_t i=0;i<si.size();++i)si[r.source_order[i]]=i;
        for(std::size_t i=0;i<di.size();++i)di[r.destination_order[i]]=i;
        struct counts {std::set<std::pair<std::uint64_t,std::uint64_t>> pairs;std::uint64_t contributions=0;};
        std::map<std::pair<std::uint64_t,std::uint64_t>,counts> tiles;
        for(std::size_t i=0;i<src.size();++i){auto a=si[src[i]],b=di[dst[i]];auto& t=tiles[{a/width,b/width}];t.pairs.emplace(a,b);++t.contributions;}
        std::vector<region> next;
        for(const auto& [key,t]:tiles) {
            auto capacity=std::min<std::uint64_t>(width,si.size()-key.first*width)*std::min<std::uint64_t>(width,di.size()-key.second*width);
            next.push_back({key.first,key.second,t.pairs.size(),t.contributions,capacity,
                double(t.pairs.size())/capacity>=threshold?region_kind::dense_nomination:region_kind::sparse});
        }
        out=std::move(next);return status::success;
    }catch(const std::bad_alloc&){return status::allocation_failure;}catch(...){return status::invalid_strategy;}
}
status repair(const problem& previous,const realization& accepted,const problem& current,const two_sided& strategy,
              realization& out,repair_report& report) noexcept {
    try {
        auto s=validate(previous,accepted);if(s!=status::success)return s;
        const auto& a=previous.operation.topology;const auto& b=current.operation.topology;
        if(!ex::same_identity(a.identity,b.identity)||!compute::operation::nf1::same_axis(a.source.identity,b.source.identity)
           ||!compute::operation::nf1::same_axis(a.destination.identity,b.destination.identity)
           ||a.source.extent!=b.source.extent||a.destination.extent!=b.destination.extent
           ||previous.state.extent!=current.state.extent||!compute::operation::nf1::same_axis(previous.state.identity,current.state.identity))return status::unsupported_projection;
        realization next;s=propose(current,strategy,next);if(s!=status::success)return s;
        repair_report stats;std::set<std::uint64_t> touched_source,touched_destination;
        auto touch=[&](const auto& job){for(const auto& argument:job.arguments)touched_source.insert(argument.index);for(const auto& output:job.outputs)touched_destination.insert(output.index);};
        std::vector<bool> unchanged(current.work.size());
        for(const auto& old:previous.work) {
            auto it=std::find_if(current.work.begin(),current.work.end(),[&](const auto& job){return same(old.instance,job.instance);});
            if(it==current.work.end()){++stats.removed_operations;touch(old);}
            else if(same_job(old,*it)){++stats.retained_operations;unchanged[it-current.work.begin()]=true;}
            else {++stats.changed_operations;touch(old);touch(*it);}
        }
        for(const auto& job:current.work)if(std::none_of(previous.work.begin(),previous.work.end(),[&](const auto& old){return same(old.instance,job.instance);})) {++stats.added_operations;touch(job);}
        stats.contribution_order_changed=!ex::same_identity(a.logical_edge_order,b.logical_edge_order)
            ||previous.work.size()!=current.work.size();
        if(!stats.contribution_order_changed)
            for(std::size_t i=0;i<previous.work.size();++i)
                if(!same(previous.work[i].instance,current.work[i].instance))stats.contribution_order_changed=true;
        // Declaration order determines flattened coefficient/contribution addressing,
        // even when every stable job is unchanged and retained independently.
        bool changed=stats.changed_operations||stats.added_operations||stats.removed_operations||stats.contribution_order_changed;
        if(b.epoch.value<a.epoch.value||(changed&&b.epoch.value<=a.epoch.value))return status::stale_structure;
        preserve(next.source_order,accepted.source_order,touched_source);preserve(next.destination_order,accepted.destination_order,touched_destination);
        next.state_order=accepted.state_order;
        auto proposed_work=next.operation_order;next.operation_order.clear();
        for(auto oldindex:accepted.operation_order) {
            auto it=std::find_if(current.work.begin(),current.work.end(),[&](const auto& job){return same(previous.work[oldindex].instance,job.instance);});
            if(it!=current.work.end()&&unchanged[it-current.work.begin()])next.operation_order.push_back(it-current.work.begin());
        }
        for(auto i:proposed_work)if(!unchanged[i])next.operation_order.push_back(i);
        for(std::size_t i=0;i<next.source_order.size();++i)stats.moved_source_rows+=next.source_order[i]!=accepted.source_order[i];
        for(std::size_t i=0;i<next.destination_order.size();++i)stats.moved_destination_rows+=next.destination_order[i]!=accepted.destination_order[i];
        const auto count=next.source_order.size()+next.destination_order.size()+next.state_order.size()+next.operation_order.size()+next.contribution_order.size();
        if(count>UINT64_MAX/sizeof(std::uint64_t))return status::capacity;
        stats.publication_bytes=count*sizeof(std::uint64_t);
        s=validate(current,next);if(s!=status::success)return s;
        out=std::move(next);report=stats;return status::success;
    }catch(const std::bad_alloc&){return status::allocation_failure;}catch(...){return status::invalid_strategy;}
}
selection choose(const cost_sample& old,const cost_sample& next,const horizon_policy& policy,bool valid) noexcept {
    const std::array<double,17> values{old.preparation_ns,old.migration_ns,old.publication_ns,old.forward_ns,old.input_vjp_ns,
      old.jvp_ns,old.expected_repair_ns,next.preparation_ns,next.migration_ns,next.publication_ns,next.forward_ns,
      next.input_vjp_ns,next.jvp_ns,next.expected_repair_ns,policy.vjp_weight,policy.jvp_weight,policy.hysteresis_ns};
    for(auto v:values)if(!std::isfinite(v)||v<0)return selection::invalid_cost;
    if(!valid)return selection::required_repair;
    if(policy.uses<policy.minimum_uses)return selection::retain;
    long double old_use=static_cast<long double>(old.forward_ns)+static_cast<long double>(policy.vjp_weight)*old.input_vjp_ns+static_cast<long double>(policy.jvp_weight)*old.jvp_ns+old.expected_repair_ns;
    long double new_use=static_cast<long double>(next.forward_ns)+static_cast<long double>(policy.vjp_weight)*next.input_vjp_ns+static_cast<long double>(policy.jvp_weight)*next.jvp_ns+next.expected_repair_ns;
    long double transition=static_cast<long double>(next.preparation_ns)+next.migration_ns+next.publication_ns+policy.hysteresis_ns;
    return policy.uses*(old_use-new_use)>transition?selection::migrate:selection::retain;
}
status prepare_routes(const problem& p,const realization& r,ex::structure_id response,relation_routes& out) noexcept {
    try {
        if(!ex::valid_identity(response)||ex::same_identity(response,p.operation.topology.identity))return status::invalid_identity;
        relation_routes next;auto s=lower_host_relation(p,r,next.forward);if(s!=status::success)return s;
        std::vector<std::uint64_t> src,dst;s=endpoints(p,src,dst);if(s!=status::success)return s;
        std::vector<std::uint64_t> si(r.source_order.size()),di(r.destination_order.size());
        for(std::size_t i=0;i<si.size();++i)si[r.source_order[i]]=i;
        for(std::size_t i=0;i<di.size();++i)di[r.destination_order[i]]=i;
        for(std::size_t i=0;i<src.size();++i){src[i]=si[src[i]];dst[i]=di[dst[i]];}
        auto op=next.forward.native.descriptor();std::swap(op.topology.source,op.topology.destination);op.topology.identity=response;
        if(!next.input_vjp.prepare(op,{dst,src}))return status::unsupported_semantics;
        next.response_input=op.topology.source;next.response_output=op.topology.destination;
        out=std::move(next);return status::success;
    }catch(const std::bad_alloc&){return status::allocation_failure;}catch(...){return status::provider_failure;}
}
} // namespace cellerator::packing::strategies
