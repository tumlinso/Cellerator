#include <Cellerator/compiler/profile/define_named_profile_environments_and_alternatives_v1.hh>

#include <cassert>
#include <cstring>

int main() {
    using namespace cellerator::compiler::profile::v1;
    const named_profile_state_v1 states[] = {
        {{1u, 1u}, {11u, 1u}, {21u, 1u}, {}, 0.5, named_profile_state_flag_baseline, 0u},
        {{2u, 2u}, {12u, 2u}, {22u, 2u}, {32u, 2u}, 0.3, named_profile_state_flag_activated, 0u},
        {{3u, 3u}, {13u, 3u}, {23u, 3u}, {33u, 3u}, 0.15, named_profile_state_flag_perturbed, 0u},
        {{4u, 4u}, {14u, 4u}, {}, {34u, 4u}, 0.05, named_profile_state_flag_unknown, 0u}};
    const named_profile_alias_v1 aliases[] = {{{101u, 1u}, states[1].state}};
    named_profile_environment_v1 environment{};
    environment.identity = {99u, 100u};
    environment.default_state = states[0].state;
    environment.states = states;
    environment.state_count = 4u;
    environment.aliases = aliases;
    environment.alias_count = 1u;
    assert(validate_named_profile_environment_v1(environment)
           == named_profile_environment_status_v1::ok);
    assert(default_named_profile_state_v1(environment)->flags
           == named_profile_state_flag_baseline);
    assert(find_named_profile_state_v1(environment, aliases[0].alias)->state.low == 2u);

    named_profile_state_v1 roundtrip[4]{};
    std::memcpy(roundtrip, states, sizeof(states));
    environment.states = roundtrip;
    assert(validate_named_profile_environment_v1(environment)
           == named_profile_environment_status_v1::ok);
    roundtrip[3].prior_weight = -1.0;
    assert(validate_named_profile_environment_v1(environment)
           == named_profile_environment_status_v1::invalid_weight);
}
