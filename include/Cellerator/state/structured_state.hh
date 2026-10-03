#pragma once
#include <Cellerator/execution/identity.hh>
#include <algorithm>
#include <cstdint>
#include <span>
#include <type_traits>

namespace cellerator::state {
namespace ex = execution;
struct actor_tag;
struct readout_tag;
using actor_id = ex::persistent_identity<actor_tag>;
using readout_id = ex::persistent_identity<readout_tag>;
enum class state_kind { scalar_patch, actor_private };
enum class activity { unknown, inactive, active };
enum class status {
    success, invalid_identity, invalid_layout, invalid_support, invalid_binding,
    stale_generation, stale_incarnation, incompatible_replica, alias,
    unsupported_layout, unsupported_readout
};
struct coordinate {
    actor_id actor{};
    ex::domain_id private_domain{}; // zero for a scalar patch; no latent axis inferred
    std::uint64_t local_slot = 0, incarnation = 0;
};
struct generations {
    ex::structure_id structure{};
    ex::structure_epoch epoch{};
    ex::value_generation values{}, activity{}, parameters{};
};
struct actor_layout {
    actor_id actor{};
    ex::domain_id private_domain{};
};
struct structured_layout {
    state_kind kind = state_kind::scalar_patch;
    ex::persistent_axis_identity actors{};
    std::span<const actor_layout> rows{};
    // Canonical storage has row_offsets[i+1]-row_offsets[i] private coordinates.
    // Scalar patches have width one and no private domain.
    std::span<const std::uint64_t> row_offsets{};
};
enum class support_kind {
    measurement_detection, primal_dependency, derivative_dependency,
    output_contribution, source_interval, candidate_nomination
};
struct support_universe {
    ex::domain_id identity{};
    ex::structure_id structure{};
    ex::structure_epoch epoch{};
    support_kind kind = support_kind::primal_dependency;
    std::uint64_t extent = 0;
};
struct support_view {
    support_universe universe{};
    std::span<const std::uint64_t> indices{}; // strictly increasing canonical slots
    bool contains(std::uint64_t index) const noexcept {
        return std::binary_search(indices.begin(), indices.end(), index);
    }
};
struct physical_binding {
    coordinate logical{};
    std::uint64_t canonical_offset = 0;
    bool capacity = true;
    activity runtime_activity = activity::unknown;
    bool residual = false;
};
bool same_coordinate(const coordinate&, const coordinate&) noexcept;
bool same_generations(const generations&, const generations&) noexcept;
bool same_universe(const support_universe&, const support_universe&) noexcept;
status validate_support(const support_view&) noexcept;
status validate_layout(const structured_layout&, std::span<const coordinate>) noexcept;
// Uniform row-major shape describes storage/algebra, never a shared biological
// coordinate meaning across actors. Ragged layouts explicitly reject this view.
status uniform_shape(const structured_layout&, std::uint64_t& rows,
                     std::uint64_t& columns) noexcept;

template<class T> struct owner_view {
    std::span<const T> values{};
    std::span<const coordinate> coordinates{};
    const generations* current = nullptr;
};
enum class readout_kind { linear, explicit_observable, nonlinear };
template<class T> struct readout_term { std::uint64_t canonical_offset; T weight; };
template<class T> struct readout {
    readout_id role{};
    actor_id actor{};
    readout_kind kind = readout_kind::linear;
    std::span<const readout_term<T>> terms{};
};
namespace detail {
template<class A, class B> bool overlaps(std::span<A> a, std::span<B> b) noexcept {
    if (a.empty() || b.empty()) return false;
    const auto x = reinterpret_cast<std::uintptr_t>(a.data());
    const auto y = reinterpret_cast<std::uintptr_t>(b.data());
    return x <= y ? y-x < a.size_bytes() : x-y < b.size_bytes();
}
}
// Borrowed host views only. Owner values/metadata/maps/support/readouts outlive
// calls; external owner serializes access and advances the matching native
// generations whenever borrowed values or metadata change. Rebind after any
// generation change. No allocation, publication, optimizer or stream ownership.
// All metadata and aliases are checked before output writes. Zero values retain
// structural support. Gather does not infer activity or mask stored values.
template<class T> class structured_state_view {
    static_assert(std::is_same_v<T,float> || std::is_same_v<T,double>);
    owner_view<T> owner_;
    structured_layout layout_;
    std::span<const physical_binding> bindings_;
    support_view support_; // structural primal support; other tagged sets remain separate
    generations saved_{};
    bool bad_output(std::span<T> out) const noexcept {
        return (out.size() && !out.data()) || detail::overlaps(owner_.values,out)
            || detail::overlaps(owner_.coordinates,out)
            || detail::overlaps(layout_.rows,out)
            || detail::overlaps(layout_.row_offsets,out)
            || detail::overlaps(bindings_,out) || detail::overlaps(support_.indices,out)
            || (owner_.current && detail::overlaps(std::span(owner_.current,1),out));
    }
public:
    structured_state_view(owner_view<T> owner, structured_layout layout,
                          std::span<const physical_binding> bindings, support_view support)
        : owner_(owner), layout_(layout), bindings_(bindings), support_(support),
          saved_(owner.current ? *owner.current : generations{}) {}
    status validate() const noexcept {
        if (!owner_.current || owner_.values.size()!=owner_.coordinates.size()
            || (owner_.values.size() && !owner_.values.data())) return status::invalid_binding;
        if (!same_generations(saved_,*owner_.current)) return status::stale_generation;
        if (!ex::valid_identity(saved_.structure) || !saved_.epoch.value
            || !saved_.values.value || !saved_.activity.value || !saved_.parameters.value)
            return status::invalid_identity;
        auto s=validate_layout(layout_,owner_.coordinates);
        if (s!=status::success) return s;
        s=validate_support(support_);
        if (s!=status::success) return s;
        if (support_.universe.kind!=support_kind::primal_dependency
            || support_.universe.extent!=owner_.values.size()
            || !ex::same_identity(support_.universe.structure,saved_.structure)
            || support_.universe.epoch.value!=saved_.epoch.value) return status::invalid_support;
        if (bindings_.size() && !bindings_.data()) return status::invalid_binding;
        for (std::size_t i=0;i<bindings_.size();++i) {
            const auto& b=bindings_[i];
            if (b.canonical_offset>=owner_.coordinates.size()) return status::invalid_binding;
            if (!same_coordinate(b.logical,owner_.coordinates[b.canonical_offset])) return status::stale_incarnation;
            if ((support_.contains(b.canonical_offset) && !b.capacity)
                || (b.runtime_activity==activity::active && !support_.contains(b.canonical_offset))
                || (b.runtime_activity!=activity::active && b.runtime_activity!=activity::inactive
                    && b.runtime_activity!=activity::unknown)) return status::invalid_binding;
            for (std::size_t j=0;j<i;++j) if (b.canonical_offset==bindings_[j].canonical_offset
                && (b.capacity!=bindings_[j].capacity || b.runtime_activity!=bindings_[j].runtime_activity
                    || b.residual!=bindings_[j].residual)) return status::incompatible_replica;
        }
        return status::success;
    }
    status gather(std::span<T> output) const noexcept {
        auto s=validate(); if(s!=status::success) return s;
        if(output.size()!=bindings_.size()) return status::invalid_binding;
        if(bad_output(output)) return status::alias;
        for(std::size_t i=0;i<output.size();++i) output[i]=owner_.values[bindings_[i].canonical_offset];
        return status::success;
    }
    // Explicit canonical accumulation sums cotangents from physical replicas.
    // Caller owns destination; this method overwrites it without updating owner.
    status pullback(std::span<const T> cotangents, std::span<T> output) const noexcept {
        auto s=validate(); if(s!=status::success) return s;
        if(cotangents.size()!=bindings_.size() || output.size()!=owner_.values.size()
            || (cotangents.size() && !cotangents.data())) return status::invalid_binding;
        if(bad_output(output) || detail::overlaps(cotangents,output)) return status::alias;
        std::fill(output.begin(),output.end(),T{});
        for(std::size_t i=0;i<cotangents.size();++i) output[bindings_[i].canonical_offset]+=cotangents[i];
        return status::success;
    }
    status readout_values(std::span<const readout<T>> declarations, std::span<T> output) const noexcept {
        auto s=validate(); if(s!=status::success) return s;
        if(output.size()!=declarations.size() || (declarations.size() && !declarations.data()))
            return status::invalid_binding;
        if(bad_output(output) || detail::overlaps(declarations,output)) return status::alias;
        for(std::size_t i=0;i<declarations.size();++i) {
            const auto& r=declarations[i];
            if(!ex::valid_identity(r.role) || !ex::valid_identity(r.actor)) return status::invalid_identity;
            bool known=false;
            for(const auto& row:layout_.rows) known|=ex::same_identity(row.actor,r.actor);
            if(!known) return status::invalid_identity;
            for(std::size_t j=0;j<i;++j) if(ex::same_identity(declarations[j].role,r.role)) return status::invalid_identity;
            if(r.kind!=readout_kind::linear && r.kind!=readout_kind::explicit_observable)
                return status::unsupported_readout;
            if(r.terms.size() && !r.terms.data()) return status::invalid_binding;
            if(detail::overlaps(r.terms,output)) return status::alias;
            if(r.kind==readout_kind::explicit_observable &&
                (r.terms.size()!=1 || r.terms[0].weight!=T(1))) return status::invalid_binding;
            for(const auto& term:r.terms) if(term.canonical_offset>=owner_.values.size()
                || !ex::same_identity(owner_.coordinates[term.canonical_offset].actor,r.actor))
                return status::invalid_binding;
        }
        for(std::size_t i=0;i<declarations.size();++i) {
            if(declarations[i].kind==readout_kind::explicit_observable) {
                output[i]=owner_.values[declarations[i].terms[0].canonical_offset];
                continue;
            }
            T value{};
            for(const auto& term:declarations[i].terms) value+=term.weight*owner_.values[term.canonical_offset];
            output[i]=value;
        }
        return status::success;
    }
    const structured_layout& layout() const noexcept { return layout_; }
    const support_view& support() const noexcept { return support_; }
};
} // namespace cellerator::state
