#pragma once

#include <cstdint>
#include <type_traits>

namespace cellerator::compiler::profile::v1 {

inline constexpr std::uint32_t named_profile_environment_schema_version_v1 = 1u;

struct profile_state_identity_v1 {
    std::uint64_t low = 0u;
    std::uint64_t high = 0u;
};

enum named_profile_state_flags_v1 : std::uint32_t {
    named_profile_state_flag_none = 0u,
    named_profile_state_flag_baseline = 1u << 0u,
    named_profile_state_flag_activated = 1u << 1u,
    named_profile_state_flag_perturbed = 1u << 2u,
    named_profile_state_flag_unknown = 1u << 3u
};

// Names and conditions use stable compiler symbol identities. They do not
// store path strings or duplicate the program IR to which an environment binds.
struct named_profile_state_v1 {
    profile_state_identity_v1 state{};
    profile_state_identity_v1 name{};
    profile_state_identity_v1 evidence{};
    profile_state_identity_v1 branch_condition{};
    double prior_weight = 0.0;
    std::uint32_t flags = named_profile_state_flag_none;
    std::uint32_t reserved = 0u;
};

struct named_profile_alias_v1 {
    profile_state_identity_v1 alias{};
    profile_state_identity_v1 state{};
};

struct named_profile_environment_v1 {
    std::uint32_t schema_version = named_profile_environment_schema_version_v1;
    std::uint32_t reserved = 0u;
    profile_state_identity_v1 identity{};
    profile_state_identity_v1 default_state{};
    const named_profile_state_v1 *states = nullptr;
    std::uint32_t state_count = 0u;
    const named_profile_alias_v1 *aliases = nullptr;
    std::uint32_t alias_count = 0u;
};

enum class named_profile_environment_status_v1 : std::uint8_t {
    ok = 0u,
    invalid_argument,
    unsupported_schema,
    duplicate_state,
    duplicate_name,
    invalid_weight,
    default_not_found,
    dangling_alias,
    duplicate_alias
};

named_profile_environment_status_v1 validate_named_profile_environment_v1(
    const named_profile_environment_v1 &environment) noexcept;
const named_profile_state_v1 *find_named_profile_state_v1(
    const named_profile_environment_v1 &environment,
    profile_state_identity_v1 name_or_alias) noexcept;
const named_profile_state_v1 *default_named_profile_state_v1(
    const named_profile_environment_v1 &environment) noexcept;

static_assert(std::is_standard_layout_v<named_profile_state_v1>);
static_assert(std::is_trivially_copyable_v<named_profile_state_v1>);
static_assert(std::is_standard_layout_v<named_profile_alias_v1>);
static_assert(std::is_trivially_copyable_v<named_profile_alias_v1>);

}  // namespace cellerator::compiler::profile::v1
