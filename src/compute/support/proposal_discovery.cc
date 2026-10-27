#include <Cellerator/compute/support/proposal_discovery.hh>

#include <limits>

namespace cellerator::compute::support {
namespace {

std::uint64_t next_random(std::uint64_t &state) noexcept {
    state += 0x9e3779b97f4a7c15ull;
    std::uint64_t value = state;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31u);
}

bool proposal_equal(const support_proposal_v1 &a,
                    const support_proposal_v1 &b) noexcept {
    if (a.mechanism_id != b.mechanism_id
        || a.participant_count != b.participant_count)
        return false;
    for (std::uint32_t i = 0u; i < a.participant_count; ++i)
        if (a.participant_ids[i] != b.participant_ids[i]) return false;
    return true;
}

bool contains(const std::uint64_t *values, std::uint64_t count,
              std::uint64_t value) noexcept {
    for (std::uint64_t i = 0u; i < count; ++i)
        if (values[i] == value) return true;
    return false;
}

bool same_as_known(const support_proposal_v1 &candidate,
                   const proposal_discovery_input_v1 &input) noexcept {
    for (std::uint64_t i = 0u; i < input.known_support_count; ++i)
        if (proposal_equal(candidate, input.known_support[i])) return true;
    return false;
}

bool choose_unused_entity(const std::uint64_t *pool, std::uint64_t pool_count,
                          const support_proposal_v1 &candidate,
                          std::uint32_t slot, std::uint64_t random,
                          std::uint64_t *selected) noexcept {
    std::uint64_t available = 0u;
    for (std::uint64_t i = 0u; i < pool_count; ++i) {
        bool used = false;
        for (std::uint32_t j = 0u; j < slot; ++j)
            if (candidate.participant_ids[j] == pool[i]) used = true;
        if (!used) ++available;
    }
    if (available == 0u) return false;
    auto rank = random % available;
    for (std::uint64_t i = 0u; i < pool_count; ++i) {
        bool used = false;
        for (std::uint32_t j = 0u; j < slot; ++j)
            if (candidate.participant_ids[j] == pool[i]) used = true;
        if (used) continue;
        if (rank == 0u) {
            *selected = pool[i];
            return true;
        }
        --rank;
    }
    return false;
}

bool multiply_u64(std::uint64_t left, std::uint64_t right,
                  std::uint64_t *result) noexcept {
    if (right != 0u && left > std::numeric_limits<std::uint64_t>::max() / right)
        return false;
    *result = left * right;
    return true;
}

std::uint64_t unique_known_count(
    const proposal_discovery_input_v1 &input) noexcept {
    std::uint64_t count = 0u;
    for (std::uint64_t i = 0u; i < input.known_support_count; ++i) {
        bool duplicate = false;
        for (std::uint64_t j = 0u; j < i; ++j)
            if (proposal_equal(input.known_support[i], input.known_support[j])) {
                duplicate = true;
                break;
            }
        if (!duplicate) ++count;
    }
    return count;
}

} // namespace

proposal_space_bounds_v1 calculate_proposal_space_bounds_v1(
    std::uint64_t entity_count, std::uint64_t mechanism_count,
    std::uint32_t participant_count) noexcept {
    proposal_space_bounds_v1 result{};
    if (entity_count == 0u || mechanism_count == 0u
        || participant_count == 0u || participant_count > 3u
        || entity_count < participant_count)
        return result;
    result.valid = true;
    std::uint64_t upper = mechanism_count;
    for (std::uint32_t i = 0u; i < participant_count; ++i) {
        if (!multiply_u64(upper, entity_count, &upper)) {
            upper = std::numeric_limits<std::uint64_t>::max();
            result.upper_bound_saturated = true;
            break;
        }
    }
    result.candidate_space_upper_bound = upper;
    std::uint64_t exact = mechanism_count;
    for (std::uint32_t i = 0u; i < participant_count; ++i) {
        if (!multiply_u64(exact, entity_count - i, &exact)) {
            result.exact_candidate_count = 0u;
            return result;
        }
    }
    result.exact_candidate_count = exact;
    result.exact_count_representable = true;
    return result;
}

proposal_status_v1 propose_support_candidates_v1(
    const proposal_discovery_input_v1 &input, std::uint64_t seed,
    std::uint64_t proposal_budget, std::uint64_t attempt_budget,
    support_proposal_v1 *output, std::uint64_t output_capacity,
    proposal_discovery_result_v1 *result) noexcept {
    if (result == nullptr) return proposal_status_v1::invalid_argument;
    *result = {};
    if (input.reserved != 0u || input.participant_count < 1u
        || input.participant_count > 3u
        || (input.entity_count != 0u && input.entity_ids == nullptr)
        || (input.mechanism_count != 0u && input.mechanism_ids == nullptr)
        || (input.known_support_count != 0u && input.known_support == nullptr)
        || (input.posting_seed_entity_count != 0u
            && input.posting_seed_entity_ids == nullptr)
        || (input.coactivation_count != 0u && input.recent_coactivations == nullptr)
        || input.entity_count < input.participant_count
        || input.mechanism_count == 0u)
        return proposal_status_v1::invalid_argument;
    for (std::uint64_t i = 0u; i < input.entity_count; ++i) {
        if (input.entity_ids[i] == 0u) return proposal_status_v1::invalid_input;
        for (std::uint64_t j = 0u; j < i; ++j)
            if (input.entity_ids[i] == input.entity_ids[j])
                return proposal_status_v1::invalid_input;
    }
    for (std::uint64_t i = 0u; i < input.mechanism_count; ++i) {
        if (input.mechanism_ids[i] == 0u) return proposal_status_v1::invalid_input;
        for (std::uint64_t j = 0u; j < i; ++j)
            if (input.mechanism_ids[i] == input.mechanism_ids[j])
                return proposal_status_v1::invalid_input;
    }
    for (std::uint64_t i = 0u; i < input.posting_seed_entity_count; ++i) {
        const auto entity = input.posting_seed_entity_ids[i];
        if (!contains(input.entity_ids, input.entity_count, entity))
            return proposal_status_v1::invalid_input;
        for (std::uint64_t j = 0u; j < i; ++j)
            if (input.posting_seed_entity_ids[j] == entity)
                return proposal_status_v1::invalid_input;
    }
    for (std::uint64_t i = 0u; i < input.known_support_count; ++i) {
        const auto &known = input.known_support[i];
        if (known.reserved != 0u
            || known.participant_count != input.participant_count
            || !contains(input.mechanism_ids, input.mechanism_count,
                         known.mechanism_id))
            return proposal_status_v1::invalid_input;
        for (std::uint32_t j = 0u; j < known.participant_count; ++j)
            if (!contains(input.entity_ids, input.entity_count,
                          known.participant_ids[j]))
                return proposal_status_v1::invalid_input;
        for (std::uint32_t j = 0u; j < known.participant_count; ++j)
            for (std::uint32_t k = 0u; k < j; ++k)
                if (known.participant_ids[j] == known.participant_ids[k])
                    return proposal_status_v1::invalid_input;
    }
    for (std::uint64_t i = 0u; i < input.coactivation_count; ++i) {
        const auto &pair = input.recent_coactivations[i];
        if (pair.first_entity_id == 0u || pair.second_entity_id == 0u
            || pair.first_entity_id == pair.second_entity_id
            || !contains(input.entity_ids, input.entity_count, pair.first_entity_id)
            || !contains(input.entity_ids, input.entity_count, pair.second_entity_id))
            return proposal_status_v1::invalid_input;
    }
    const auto bounds = calculate_proposal_space_bounds_v1(input.entity_count,
        input.mechanism_count, input.participant_count);
    result->candidate_space_upper_bound = bounds.candidate_space_upper_bound;
    result->candidate_space_upper_bound_saturated = bounds.upper_bound_saturated;
    const auto known_count = unique_known_count(input);
    result->missed_candidate_count_exact = bounds.exact_count_representable
        && known_count <= bounds.exact_candidate_count;
    if (result->missed_candidate_count_exact)
        result->missed_candidate_lower_bound = bounds.exact_candidate_count - known_count;
    result->next_seed = seed;
    if (output_capacity < proposal_budget) {
        result->required = proposal_budget;
        return proposal_status_v1::output_overflow;
    }
    if (proposal_budget != 0u && output == nullptr)
        return proposal_status_v1::invalid_capacity;
    auto rng = seed;
    while (result->written < proposal_budget && result->attempts < attempt_budget) {
        ++result->attempts;
        support_proposal_v1 candidate{};
        candidate.mechanism_id = input.mechanism_ids[
            next_random(rng) % input.mechanism_count];
        candidate.participant_count = input.participant_count;
        const bool prefer_postings = input.posting_seed_entity_count != 0u
            && (next_random(rng) & 1u) != 0u;

        bool used_coactivation = false;
        if (input.participant_count >= 2u && input.coactivation_count != 0u
            && (next_random(rng) & 1u) != 0u) {
            const auto &pair = input.recent_coactivations[
                next_random(rng) % input.coactivation_count];
            candidate.participant_ids[0] = pair.first_entity_id;
            candidate.participant_ids[1] = pair.second_entity_id;
            used_coactivation = true;
        }
        if (!used_coactivation) {
            const auto *pool = prefer_postings ? input.posting_seed_entity_ids
                                                : input.entity_ids;
            const auto pool_count = prefer_postings ? input.posting_seed_entity_count
                                                    : input.entity_count;
            choose_unused_entity(pool, pool_count, candidate, 0u,
                next_random(rng), &candidate.participant_ids[0]);
        }
        const std::uint32_t start = used_coactivation ? 2u : 1u;
        for (std::uint32_t slot = start; slot < input.participant_count; ++slot) {
            const auto *pool = prefer_postings ? input.posting_seed_entity_ids
                                                : input.entity_ids;
            const auto pool_count = prefer_postings ? input.posting_seed_entity_count
                                                    : input.entity_count;
            if (!choose_unused_entity(pool, pool_count, candidate, slot,
                    next_random(rng), &candidate.participant_ids[slot]))
                choose_unused_entity(input.entity_ids, input.entity_count,
                    candidate, slot, next_random(rng),
                    &candidate.participant_ids[slot]);
        }
        bool duplicate = false;
        if (same_as_known(candidate, input)) duplicate = true;
        for (std::uint64_t i = 0u; i < result->written; ++i)
            if (proposal_equal(output[i], candidate)) {
                duplicate = true;
                break;
            }
        if (!duplicate) output[result->written++] = candidate;
    }
    result->required = result->written;
    result->next_seed = rng;
    if (result->missed_candidate_count_exact
        && result->written <= result->missed_candidate_lower_bound)
        result->missed_candidate_lower_bound -= result->written;
    else {
        result->missed_candidate_lower_bound = 0u;
        result->missed_candidate_count_exact = false;
    }
    if (result->candidate_space_upper_bound != 0u
        && !result->candidate_space_upper_bound_saturated)
        result->sampled_coverage_lower_bound = static_cast<double>(result->written)
            / static_cast<double>(result->candidate_space_upper_bound);
    return proposal_status_v1::success;
}

} // namespace cellerator::compute::support
