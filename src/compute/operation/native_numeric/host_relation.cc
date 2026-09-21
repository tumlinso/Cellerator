#include <Cellerator/compute/operation/native_numeric/host_relation.hh>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace cellerator::compute::native_numeric {
namespace {
using code = relation::status_code;
bool axis_matches(const relation::axis_descriptor& x,const relation::axis_descriptor& y) noexcept {
    return x.extent==y.extent
        && execution::validate_persistent_axis_identity(y.identity)==execution::biological_validation_code::ok
        && execution::same_identity(x.identity.domain,y.identity.domain)
        && execution::same_identity(x.identity.order,y.identity.order)
        && execution::same_identity(x.identity.geometry,y.identity.geometry)
        && execution::same_identity(x.identity.partition,y.identity.partition);
}
bool overlaps(const void* a,std::size_t an,const void* b,std::size_t bn) noexcept {
    if(!an || !bn) return false;
    const auto av=reinterpret_cast<std::uintptr_t>(a),bv=reinterpret_cast<std::uintptr_t>(b);
    // Difference formulation cannot overflow while checking half-open intervals.
    return av<=bv ? bv-av<an : av-bv<bn;
}
}
host_relation::host_relation(host_relation&& other) noexcept { *this=std::move(other); }
host_relation& host_relation::operator=(host_relation&& other) noexcept {
    if(this!=&other){
        prepared_=other.prepared_;descriptor_=other.descriptor_;
        offsets_=std::move(other.offsets_);edges_=std::move(other.edges_);sources_=std::move(other.sources_);
        other.prepared_=false;
    }
    return *this;
}
relation::status host_relation::prepare(const relation::operation_descriptor& op,host_topology topology) noexcept {
    auto valid=relation::validate(op);if(!valid)return valid;
    if(op.direction!=relation::orientation::forward || op.update!=relation::output_update::overwrite
       || op.input_output_aliasing_legal)
        return {code::unsupported_semantics,"host N01 supports forward overwrite without aliases"};
    const auto& a=op.arithmetic;
    if((a.relation_storage!=execution::numeric_type::f32 && a.relation_storage!=execution::numeric_type::f64)
        || a.input_storage!=a.relation_storage || a.multiply!=a.relation_storage
        || a.accumulation!=a.relation_storage || a.output_storage!=a.relation_storage)
        return {code::unsupported_numeric_policy,"host route requires homogeneous f32 or f64"};
    const auto& t=op.topology;
    if(topology.sources.size()!=t.edge_count || topology.destinations.size()!=t.edge_count)
        return {code::invalid_shape,"endpoint counts differ from logical topology"};
    if(t.destination.extent>=std::numeric_limits<std::size_t>::max()/sizeof(std::uint64_t)
       || t.edge_count>std::numeric_limits<std::size_t>::max()/sizeof(std::uint64_t))
        return {code::insufficient_capacity,"prepared index capacity overflows"};
    for(std::size_t i=0;i<topology.sources.size();++i)
        if(topology.sources[i]>=t.source.extent || topology.destinations[i]>=t.destination.extent)
            return {code::invalid_shape,"logical endpoint outside axis"};
    try {
        host_relation next;
        next.offsets_.assign(static_cast<std::size_t>(t.destination.extent)+1,0);
        next.sources_.assign(topology.sources.begin(),topology.sources.end());
        next.edges_.resize(topology.sources.size());
        for(auto destination:topology.destinations)++next.offsets_[destination+1];
        for(std::size_t i=1;i<next.offsets_.size();++i)next.offsets_[i]+=next.offsets_[i-1];
        auto cursor=next.offsets_;
        for(std::size_t edge=0;edge<topology.destinations.size();++edge)
            next.edges_[cursor[topology.destinations[edge]]++]=edge;
        next.descriptor_=op;next.prepared_=true;
        *this=std::move(next);
        return {};
    }catch(const std::bad_alloc&){return {code::insufficient_capacity,"host preparation allocation failed"};}
    catch(const std::length_error&){return {code::insufficient_capacity,"host preparation size exceeds container limit"};}
}

template<class T>
relation::status host_relation::execute(value_identity identity,std::span<const T> values,
        const relation::axis_descriptor& input_axis,std::span<const T> input,
        const relation::axis_descriptor& output_axis,std::span<T> output) const noexcept {
    if(!prepared_)return {code::invalid_state,"host relation is not prepared"};
    constexpr auto type=std::is_same_v<T,float>?execution::numeric_type::f32:execution::numeric_type::f64;
    const auto& op=descriptor_;const auto& t=op.topology;
    if(op.arithmetic.relation_storage!=type)return {code::unsupported_numeric_policy,"binding type differs from prepared arithmetic"};
    if(!execution::same_identity(identity.structure,t.identity) || identity.epoch.value!=t.epoch.value)
        return {code::stale_structure,"value structure or epoch differs"};
    if(!execution::same_identity(identity.edge_order,t.logical_edge_order))
        return {code::incompatible_order,"values are not in declared logical edge order"};
    if(identity.generation.value==0)return {code::stale_generation,"value generation must be nonzero"};
    if(!axis_matches(t.source,input_axis) || !axis_matches(t.destination,output_axis))
        return {code::incompatible_order,"input or output axis identity differs"};
    const auto input_count=t.source.extent*op.dense_width;
    const auto output_count=t.destination.extent*op.dense_width;
    if(values.size()!=t.edge_count || input.size()!=input_count || output.size()!=output_count)
        return {code::insufficient_capacity,"host binding element counts differ"};
    if((!values.empty()&&!values.data()) || (!input.empty()&&!input.data()) || (!output.empty()&&!output.data()))
        return {code::invalid_argument,"nonempty host binding is null"};
    if(overlaps(output.data(),output.size_bytes(),input.data(),input.size_bytes())
       || overlaps(output.data(),output.size_bytes(),values.data(),values.size_bytes()))
        return {code::unsupported_semantics,"output aliases a launch input"};
    if(op.arithmetic.nonfinite==relation::nonfinite_policy::reject){
        for(T v:values)if(!std::isfinite(v))return {code::invalid_argument,"nonfinite relation value"};
        for(T v:input)if(!std::isfinite(v))return {code::invalid_argument,"nonfinite dense input"};
    }
    // All fallible admission is complete. Stable per-destination logical edge order;
    // duplicate edges contribute independently. No allocation or topology discovery.
    for(std::size_t row=0;row<t.destination.extent;++row)
        for(std::size_t channel=0;channel<op.dense_width;++channel){
            T total=0;
            for(auto cursor=offsets_[row];cursor<offsets_[row+1];++cursor){
                const auto edge=edges_[cursor];
                const T product=values[edge]*input[sources_[edge]*op.dense_width+channel];
                total=total+product;
            }
            output[row*op.dense_width+channel]=total;
        }
    return {};
}
relation::status host_relation::run(value_identity id,std::span<const float> w,
 const relation::axis_descriptor& ia,std::span<const float> x,
 const relation::axis_descriptor& oa,std::span<float> y) const noexcept{return execute(id,w,ia,x,oa,y);}
relation::status host_relation::run(value_identity id,std::span<const double> w,
 const relation::axis_descriptor& ia,std::span<const double> x,
 const relation::axis_descriptor& oa,std::span<double> y) const noexcept{return execute(id,w,ia,x,oa,y);}
} // namespace cellerator::compute::native_numeric
