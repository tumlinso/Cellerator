#include <Cellerator/state/structured_state.hh>
#include <array>
#include <iostream>
#include <stdexcept>
namespace st=cellerator::state;
namespace ex=cellerator::execution;
int checks=0;
void require(bool b,const char* why) { ++checks; if(!b) throw std::runtime_error(why); }
void success(st::status s) { require(s==st::status::success,"expected success"); }
ex::persistent_axis_identity axis() {
    return {{ex::biological_abi_version,ex::serialized_record_kind::persistent_axis_identity,
        sizeof(ex::persistent_axis_identity)},{1,1},{2,1},{3,1},{4,1}};
}
void ragged() {
    // Two actors have distinct private coordinate meaning and widths 2 and 1.
    std::array<st::actor_layout,2> rows{{{{11,1},{51,1}},{{12,1},{52,1}}}};
    std::array<std::uint64_t,3> offsets{0,2,3};
    std::array<st::coordinate,3> ids{{{{11,1},{51,1},0,7},{{11,1},{51,1},1,3},{{12,1},{52,1},0,2}}};
    std::array<double,3> values{0,2,9};
    st::generations stamp{{100,1},{2},{3},{4},{5}};
    st::structured_layout layout{st::state_kind::actor_private,axis(),rows,offsets};
    // Primal support is distinct from detection support, even when values are zero.
    std::array<std::uint64_t,2> support_ids{0,2};
    st::support_view support{{{200,1},stamp.structure,stamp.epoch,st::support_kind::primal_dependency,3},support_ids};
    auto detection=support.universe; detection.kind=st::support_kind::measurement_detection;
    require(!st::same_universe(detection,support.universe),"support tags are different meanings");
    auto another=support.universe; ++another.identity.low;
    require(!st::same_universe(another,support.universe),"support universes distinct");
    another=support.universe; ++another.epoch.value;
    require(!st::same_universe(another,support.universe),"support epoch distinct");
    std::array<st::physical_binding,4> maps{{{ids[2],2,true,st::activity::active,false},
        {ids[0],0,true,st::activity::inactive,false},{ids[0],0,true,st::activity::inactive,false},
        {ids[1],1,true,st::activity::inactive,true}}};
    st::owner_view<double> owner{values,ids,&stamp};
    auto bind=[&] { return st::structured_state_view<double>(owner,layout,maps,support); };
    auto view=bind(); success(view.validate());
    support.universe.kind=st::support_kind::measurement_detection;
    require(bind().validate()==st::status::invalid_support,"measurement support cannot become structural dependency support");
    support.universe.kind=st::support_kind::primal_dependency;
    std::array<double,4> gathered{}; success(view.gather(gathered));
    require(gathered==std::array<double,4>{9,0,0,2},"physical order and replicated zero supported state");
    std::array<double,4> cotangents{5,2,3,7}; std::array<double,3> gradient{};
    success(view.pullback(cotangents,gradient));
    require(gradient==std::array<double,3>{5,7,5},"replica cotangents sum to canonical coordinates");
    require(values==std::array<double,3>{0,2,9},"pullback never publishes owner values");
    // Readout semantics belong to the named actor, independently of physical replicas.
    std::array<st::readout_term<double>,2> linear_terms{{{0,3},{1,-2}}};
    std::array<st::readout_term<double>,1> observable{{{2,1}}};
    std::array<st::readout<double>,2> reads{{{{71,1},rows[0].actor,st::readout_kind::linear,linear_terms},
        {{72,1},rows[1].actor,st::readout_kind::explicit_observable,observable}}};
    std::array<double,2> measured{}; success(view.readout_values(reads,measured));
    require(measured==std::array<double,2>{-4,9},"readout meaning independent of layout/gather");
    linear_terms[1].canonical_offset=2; measured.fill(888);
    require(view.readout_values(reads,measured)==st::status::invalid_binding && measured[0]==888,"cross-actor readout rejected before any write");
    linear_terms[1].canonical_offset=1; reads[1].kind=st::readout_kind::nonlinear;
    require(view.readout_values(reads,measured)==st::status::unsupported_readout && measured[0]==888,"nonlinear readout explicitly unsupported");
    reads[1].kind=st::readout_kind::explicit_observable;
    std::uint64_t matrix_rows=999,columns=888;
    require(st::uniform_shape(layout,matrix_rows,columns)==st::status::unsupported_layout && matrix_rows==999 && columns==888,"ragged storage cannot become a uniform matrix implicitly");
    // Owner values/generations remain native borrowed state.
    values[0]=4; ++stamp.values.value;
    gathered.fill(777);
    require(view.gather(gathered)==st::status::stale_generation && gathered[0]==777,"old value generation rejected without output writes");
    view=bind(); success(view.gather(gathered)); require(gathered[1]==4,"rebound view sees owner updates");
    ++stamp.parameters.value;
    require(view.validate()==st::status::stale_generation,"parameter generation invalidates saved binding");
    view=bind(); ++stamp.activity.value;
    require(view.validate()==st::status::stale_generation,"activity generation invalidates saved binding");
    view=bind(); ++stamp.epoch.value;
    require(view.validate()==st::status::stale_generation,"structure publication invalidates old binding");
    require(bind().validate()==st::status::invalid_support,"old support universe incompatible with new structure");
    support.universe.epoch=stamp.epoch; view=bind(); success(view.validate());
    ++maps[1].logical.incarnation;
    require(bind().validate()==st::status::stale_incarnation,"stale physical incarnation rejected");
    maps[1].logical=ids[0]; maps[2].runtime_activity=st::activity::active;
    require(bind().validate()==st::status::incompatible_replica,"replicas must agree on activity");
    maps[2].runtime_activity=st::activity::inactive; maps[2].residual=true;
    require(bind().validate()==st::status::incompatible_replica,"replicas must agree on residual identity");
    maps[2].residual=false; maps[1].capacity=false;
    require(bind().validate()==st::status::invalid_binding,"supported state requires capacity");
    maps[1].capacity=true; maps[3].runtime_activity=st::activity::active;
    require(bind().validate()==st::status::invalid_binding,"activity cannot invent support");
    maps[3].runtime_activity=st::activity::inactive;
    auto duplicate=rows[1].actor; rows[1].actor=rows[0].actor;
    require(bind().validate()==st::status::invalid_identity,"duplicate actor IDs rejected"); rows[1].actor=duplicate;
    auto slot=ids[1].local_slot; ids[1].local_slot=0;
    require(bind().validate()==st::status::stale_incarnation,"duplicate owner slot rejected even with distinct incarnation"); ids[1].local_slot=slot;
    support_ids[1]=0; require(bind().validate()==st::status::invalid_support,"duplicate support members rejected"); support_ids[1]=2;
    support_ids[1]=3; require(bind().validate()==st::status::invalid_support,"support universe bounds checked"); support_ids[1]=2;
    view=bind(); require(view.pullback(cotangents,values)==st::status::alias,"owner values cannot be cotangent destination");
    require(view.gather({values.data(),3})==st::status::invalid_binding,"wrong physical extent rejected");
    require(view.pullback({gradient.data(),3},gradient)==st::status::invalid_binding,"wrong cotangent extent rejected");
    owner.current=nullptr; require(bind().validate()==st::status::invalid_binding,"missing owner generation source rejected");
}
void scalar_and_uniform() {
    std::array<st::actor_layout,2> rows{{{{21,1},{}},{{22,1},{}}}};
    std::array<std::uint64_t,3> offsets{0,1,2};
    std::array<st::coordinate,2> ids{{{{21,1},{},0,1},{{22,1},{},0,1}}};
    std::array<float,2> values{2,-3}; st::generations stamp{{100,1},{1},{1},{1},{1}};
    st::structured_layout layout{st::state_kind::scalar_patch,axis(),rows,offsets};
    std::array<st::physical_binding,2> maps{{{ids[1],1},{ids[0],0}}};
    std::array<std::uint64_t,2> members{0,1};
    st::support_view support{{{300,1},stamp.structure,stamp.epoch,st::support_kind::primal_dependency,2},members};
    st::structured_state_view<float> view({values,ids,&stamp},layout,maps,support);
    std::array<float,2> output{}; success(view.gather(output));
    require(output==std::array<float,2>{-3,2},"scalar patch retains logical identity through physical reorder");
    std::uint64_t matrix_rows{},columns{}; success(st::uniform_shape(layout,matrix_rows,columns));
    require(matrix_rows==2 && columns==1 && !ex::valid_identity(rows[0].private_domain),"scalar matrix storage creates no fictional private regulator");
    auto old_axis=layout.actors; layout.actors.order={};
    require(st::validate_layout(layout,ids)==st::status::invalid_identity,"native domain/order admission used"); layout.actors=old_axis;
    rows[0].private_domain={9,1}; require(st::validate_layout(layout,ids)==st::status::invalid_layout,"scalar patch rejects invented hidden axis");
    rows[0].private_domain={51,1}; rows[1].private_domain={52,1};
    ids[0].private_domain=rows[0].private_domain; ids[1].private_domain=rows[1].private_domain;
    layout.kind=st::state_kind::actor_private; success(st::validate_layout(layout,ids));
    success(st::uniform_shape(layout,matrix_rows,columns));
    require(matrix_rows==2 && columns==1 && !ex::same_identity(rows[0].private_domain,rows[1].private_domain),"uniform storage retains actor-local algebraic axes with distinct meaning");
    require(view.gather(values)==st::status::invalid_layout,"old scalar declaration incompatible with changed declaration");
    // Fresh actor-private view validates aliases with matching metadata.
    maps[0].logical=ids[1]; maps[1].logical=ids[0];
    st::structured_state_view<float> rebound({values,ids,&stamp},layout,maps,support);
    require(rebound.gather(values)==st::status::alias,"direct owner overlap rejected before writes");
    std::array<float,2> gradients{3,4};
    require(rebound.pullback(gradients,gradients)==st::status::alias,"cotangent alias rejected");
}
void empty_state() {
    std::array<std::uint64_t,1> offsets{0};
    st::structured_layout layout{st::state_kind::scalar_patch,axis(),{},offsets};
    st::generations stamp{{100,1},{1},{1},{1},{1}};
    st::support_view support{{{400,1},stamp.structure,stamp.epoch,st::support_kind::primal_dependency,0},{}};
    st::structured_state_view<double> view({{},{},&stamp},layout,{},support);
    success(view.validate()); success(view.gather({})); success(view.pullback({},{})); success(view.readout_values({},{}));
}
int main() try {
    ragged(); scalar_and_uniform(); empty_state();
    std::cout<<"PASS structured native host state: "<<checks<<" checks\n";
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
