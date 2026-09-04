#include <Cellerator/compiler/profile/define_named_profile_environments_and_alternatives_v1.hh>

#include <cmath>

namespace cellerator::compiler::profile::v1 {
namespace {
bool same(profile_state_identity_v1 a, profile_state_identity_v1 b) noexcept {
    return a.low == b.low && a.high == b.high;
}
bool zero(profile_state_identity_v1 a) noexcept { return a.low == 0u && a.high == 0u; }
const named_profile_state_v1 *by_state(
    const named_profile_environment_v1 &environment,
    profile_state_identity_v1 state) noexcept {
    for (std::uint32_t i = 0; i < environment.state_count; ++i)
        if (same(environment.states[i].state, state)) return &environment.states[i];
    return nullptr;
}
}  // namespace

named_profile_environment_status_v1 validate_named_profile_environment_v1(
    const named_profile_environment_v1 &environment) noexcept {
    if ((environment.state_count != 0u && environment.states == nullptr)
        || (environment.alias_count != 0u && environment.aliases == nullptr)
        || zero(environment.identity))
        return named_profile_environment_status_v1::invalid_argument;
    if (environment.schema_version != named_profile_environment_schema_version_v1)
        return named_profile_environment_status_v1::unsupported_schema;
    for (std::uint32_t i = 0; i < environment.state_count; ++i) {
        const auto &state = environment.states[i];
        if (zero(state.state) || zero(state.name))
            return named_profile_environment_status_v1::invalid_argument;
        if (!std::isfinite(state.prior_weight) || state.prior_weight < 0.0)
            return named_profile_environment_status_v1::invalid_weight;
        for (std::uint32_t j = 0; j < i; ++j) {
            if (same(state.state, environment.states[j].state))
                return named_profile_environment_status_v1::duplicate_state;
            if (same(state.name, environment.states[j].name))
                return named_profile_environment_status_v1::duplicate_name;
        }
    }
    if (by_state(environment, environment.default_state) == nullptr)
        return named_profile_environment_status_v1::default_not_found;
    for (std::uint32_t i = 0; i < environment.alias_count; ++i) {
        const auto &alias = environment.aliases[i];
        if (zero(alias.alias) || by_state(environment, alias.state) == nullptr)
            return named_profile_environment_status_v1::dangling_alias;
        for (std::uint32_t j = 0; j < i; ++j)
            if (same(alias.alias, environment.aliases[j].alias))
                return named_profile_environment_status_v1::duplicate_alias;
    }
    return named_profile_environment_status_v1::ok;
}

const named_profile_state_v1 *find_named_profile_state_v1(
    const named_profile_environment_v1 &environment,
    profile_state_identity_v1 name_or_alias) noexcept {
    for (std::uint32_t i = 0; i < environment.state_count; ++i)
        if (same(environment.states[i].name, name_or_alias))
            return &environment.states[i];
    for (std::uint32_t i = 0; i < environment.alias_count; ++i)
        if (same(environment.aliases[i].alias, name_or_alias))
            return by_state(environment, environment.aliases[i].state);
    return nullptr;
}

const named_profile_state_v1 *default_named_profile_state_v1(
    const named_profile_environment_v1 &environment) noexcept {
    return by_state(environment, environment.default_state);
}
}  // namespace cellerator::compiler::profile::v1
