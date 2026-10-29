#pragma once

#include <Cellerator/modeling/entity_layout.hh>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cellerator::modeling {

struct ExchangeLink {
    std::uint64_t identity;
    std::uint64_t cell;
    std::uint64_t compartment;
    double weight = 1.0;
};

enum class DonorPolicy { reject, limit };

// Arrays use their respective EntityLayout order, then species order.
template<class Scalar> struct ExchangeState {
    std::vector<Scalar> cells;
    std::vector<Scalar> compartments;
};

template<class Scalar> struct ExchangeResult {
    ExchangeState<Scalar> state;
    // Link-major, species-minor; positive means cell -> compartment.
    std::vector<long double> transfers;
    long double passive_dt_limit;
    bool limited;
};

// A host reference operator. Mutable amount/volume values are never stored in
// the structural incidence plan. Entity and species identity survive layout
// changes through explicit EntityLayout mapping. Floating storage is selected
// by the caller; multiply/accumulation use long double, outputs use Scalar.
class ExchangePlan final {
public:
    ExchangePlan(EntityLayout cells, EntityLayout compartments, EntityLayout species,
                 std::vector<ExchangeLink> links)
        : cells_(std::move(cells)), compartments_(std::move(compartments)),
          species_(std::move(species)), links_(std::move(links)) {
        if (species_.size() == 0)
            throw std::invalid_argument("exchange requires at least one species");
        checked_size(cells_.size(), species_.size());
        checked_size(compartments_.size(), species_.size());
        checked_size(links_.size(), species_.size());
        std::unordered_set<std::uint64_t> seen;
        for (const auto& link : links_) {
            if (!seen.insert(link.identity).second)
                throw std::invalid_argument("duplicate exchange link identity");
            if (!std::isfinite(link.weight) || link.weight < 0)
                throw std::invalid_argument("exchange weight must be finite and nonnegative");
            cell_slots_.push_back(cells_.slot(link.cell));
            compartment_slots_.push_back(compartments_.slot(link.compartment));
        }
    }

    const EntityLayout& cells() const noexcept { return cells_; }
    const EntityLayout& compartments() const noexcept { return compartments_; }
    const EntityLayout& species() const noexcept { return species_; }
    const std::vector<ExchangeLink>& links() const noexcept { return links_; }

    template<class Scalar>
    long double passive_dt_limit(const std::vector<Scalar>& cell_volumes,
                                 const std::vector<Scalar>& compartment_volumes,
                                 const std::vector<Scalar>& permeability) const {
        validate_parameters(cell_volumes, compartment_volumes, permeability);
        const auto n = species_.size();
        std::vector<long double> cd(cells_.size() * n), ed(compartments_.size() * n);
        for (std::size_t e = 0; e < links_.size(); ++e) {
            for (std::size_t s = 0; s < n; ++s) {
                const auto k = static_cast<long double>(links_[e].weight) * permeability[s];
                cd[cell_slots_[e] * n + s] += k;
                ed[compartment_slots_[e] * n + s] += k;
            }
        }
        auto bound = std::numeric_limits<long double>::infinity();
        for (std::size_t i = 0; i < cd.size(); ++i) {
            finite(cd[i]);
            if (cd[i] > 0) bound = std::min(bound, cell_volumes[i / n] / cd[i]);
        }
        for (std::size_t i = 0; i < ed.size(); ++i) {
            finite(ed[i]);
            if (ed[i] > 0) bound = std::min(bound, compartment_volumes[i / n] / ed[i]);
        }
        return bound;
    }

    // q[e,s] = dt * weight[e] * (P[s]*(cell_amount/Vc-env_amount/Ve)
    //                           + active_rate[e,s]). Active rates are signed
    // amount/time, permeability volume/time. Weights are never normalized.
    // Every request uses the original snapshot. Limiting spends only initial
    // donor material, shared proportionally among all outgoing requests.
    template<class Scalar>
    ExchangeResult<Scalar> step(
        const ExchangeState<Scalar>& state,
        const std::vector<Scalar>& cell_volumes,
        const std::vector<Scalar>& compartment_volumes,
        const std::vector<Scalar>& permeability,
        long double dt,
        const std::vector<Scalar>& active_rates = {},
        DonorPolicy donor_policy = DonorPolicy::reject,
        bool enforce_stability = true) const {
        static_assert(std::is_floating_point_v<Scalar>, "exchange storage must be floating point");
        if (!std::isfinite(dt) || dt < 0)
            throw std::invalid_argument("exchange dt must be finite and nonnegative");
        if (donor_policy != DonorPolicy::reject && donor_policy != DonorPolicy::limit)
            throw std::invalid_argument("unknown donor policy");
        const auto bound = passive_dt_limit(cell_volumes, compartment_volumes, permeability);
        if (enforce_stability && dt > bound)
            throw std::invalid_argument("exchange dt exceeds passive positivity bound");
        const auto n = species_.size();
        validate(state.cells, cells_.size() * n, false, false);
        validate(state.compartments, compartments_.size() * n, false, false);
        if (!active_rates.empty()) validate(active_rates, links_.size() * n, true, false);
        std::vector<long double> requests(links_.size() * n);
        std::vector<Sum> co(state.cells.size()), eo(state.compartments.size());
        for (std::size_t e = 0; e < links_.size(); ++e) {
            for (std::size_t s = 0; s < n; ++s) {
                const auto c = cell_slots_[e] * n + s;
                const auto v = compartment_slots_[e] * n + s;
                const auto cc = static_cast<long double>(state.cells[c]) / cell_volumes[cell_slots_[e]];
                const auto ec = static_cast<long double>(state.compartments[v]) / compartment_volumes[compartment_slots_[e]];
                const auto active = active_rates.empty() ? 0.L : static_cast<long double>(active_rates[e * n + s]);
                const auto q = dt * links_[e].weight * (permeability[s] * (cc - ec) + active);
                finite(q);
                requests[e * n + s] = q;
                if (q >= 0) co[c].add(q); else eo[v].add(-q);
            }
        }
        bool limited = false;
        const auto cs = scales(state.cells, co, donor_policy, limited);
        const auto es = scales(state.compartments, eo, donor_policy, limited);
        std::vector<Sum> ci(co.size()), ei(eo.size());
        std::fill(co.begin(), co.end(), Sum{});
        std::fill(eo.begin(), eo.end(), Sum{});
        for (std::size_t e = 0; e < links_.size(); ++e) {
            for (std::size_t s = 0; s < n; ++s) {
                const auto c = cell_slots_[e] * n + s;
                const auto v = compartment_slots_[e] * n + s;
                auto& q = requests[e * n + s];
                q *= q >= 0 ? cs[c] : es[v];
                if (q >= 0) { co[c].add(q); ei[v].add(q); }
                else { eo[v].add(-q); ci[c].add(-q); }
            }
        }
        ExchangeState<Scalar> next{apply(state.cells, co, ci), apply(state.compartments, eo, ei)};
        return {std::move(next), std::move(requests), bound, limited};
    }

private:
    // Compensated accounting prevents a large fanout from spuriously spending
    // more than the donor budget through repeated-addition roundoff.
    struct Sum {
        long double value = 0.L;
        long double correction = 0.L;
        void add(long double increment) {
            const auto adjusted = increment - correction;
            const auto next = value + adjusted;
            correction = (next - value) - adjusted;
            value = next;
        }
    };

    static void checked_size(std::size_t rows, std::size_t columns) {
        if (rows > std::numeric_limits<std::size_t>::max() / columns)
            throw std::length_error("exchange shape overflows size_t");
    }

    static void finite(long double value) {
        if (!std::isfinite(value))
            throw std::overflow_error("exchange exceeds numerical range");
    }

    template<class Scalar>
    static void validate(const std::vector<Scalar>& values, std::size_t size,
                         bool signed_values, bool positive) {
        static_assert(std::is_floating_point_v<Scalar>, "exchange storage must be floating point");
        if (values.size() != size) throw std::invalid_argument("exchange array shape mismatch");
        for (const auto value : values) {
            if (!std::isfinite(value) || (!signed_values && value < 0) || (positive && value <= 0))
                throw std::invalid_argument("invalid exchange value or volume");
        }
    }

    template<class Scalar>
    void validate_parameters(const std::vector<Scalar>& cv, const std::vector<Scalar>& ev,
                             const std::vector<Scalar>& p) const {
        validate(cv, cells_.size(), false, true);
        validate(ev, compartments_.size(), false, true);
        validate(p, species_.size(), false, false);
    }

    template<class Scalar>
    static std::vector<long double> scales(const std::vector<Scalar>& amount,
                                          const std::vector<Sum>& outgoing,
                                          DonorPolicy policy, bool& limited) {
        std::vector<long double> result(amount.size(), 1.L);
        for (std::size_t i = 0; i < amount.size(); ++i) {
            finite(outgoing[i].value);
            if (outgoing[i].value > amount[i]) {
                if (policy == DonorPolicy::reject)
                    throw std::invalid_argument("simultaneous exchange exceeds donor amount");
                result[i] = amount[i] == 0 ? 0.L : std::nextafter(amount[i] / outgoing[i].value, 0.L);
                limited = true;
            }
        }
        return result;
    }

    template<class Scalar>
    static std::vector<Scalar> apply(const std::vector<Scalar>& amount,
                                     const std::vector<Sum>& spent,
                                     const std::vector<Sum>& incoming) {
        std::vector<Scalar> result(amount.size());
        for (std::size_t i = 0; i < amount.size(); ++i) {
            const auto value = (static_cast<long double>(amount[i]) - spent[i].value) + incoming[i].value;
            finite(value);
            if (value < 0 || value > std::numeric_limits<Scalar>::max())
                throw std::overflow_error("exchange output exceeds storage range; reduce step");
            result[i] = static_cast<Scalar>(value);
        }
        return result;
    }

    EntityLayout cells_, compartments_, species_;
    std::vector<ExchangeLink> links_;
    std::vector<std::size_t> cell_slots_, compartment_slots_;
};

} // namespace cellerator::modeling
