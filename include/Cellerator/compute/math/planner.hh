#pragma once

#include "backend.hh"

#include <cstddef>

namespace cellerator::compute::math {

inline constexpr u32 math_planner_schema_version = 1u;
inline constexpr std::size_t max_spmm_backend_count = 32u;
inline constexpr std::size_t max_planner_candidate_count = 32u;

enum class backend_registration_code : u32 {
    ok = 0u,
    invalid_backend = 1u,
    duplicate_identity = 2u,
    capacity_exceeded = 3u,
    not_found = 4u
};

struct backend_registration_status {
    backend_registration_code code = backend_registration_code::ok;
    const char *message = "ok";

    constexpr explicit operator bool() const noexcept {
        return code == backend_registration_code::ok;
    }
};

// Registration is a startup/configuration operation. Callers must not mutate a
// registry concurrently with planning against it. Backends are borrowed and
// must outlive both the registry and every PreparedExecution selected from it.
class SpMMBackendRegistry {
public:
    backend_registration_status add(SpMMBackend *backend) noexcept;
    backend_registration_status remove(u64 identity) noexcept;
    void clear() noexcept;
    std::size_t size() const noexcept;
    SpMMBackend *at(std::size_t index) const noexcept;

private:
    SpMMBackend *backends_[max_spmm_backend_count]{};
    std::size_t size_ = 0u;
};

SpMMBackendRegistry &global_spmm_backend_registry() noexcept;

enum class planner_status_code : u32 {
    ok = 0u,
    invalid_argument = 1u,
    invalid_request = 2u,
    invalid_device = 3u,
    no_legal_candidate = 4u
};

struct planner_status {
    planner_status_code code = planner_status_code::ok;
    request_validation_code validation = request_validation_code::ok;
    capability_code capability = capability_code::supported;
    const char *message = "ok";

    constexpr explicit operator bool() const noexcept {
        return code == planner_status_code::ok;
    }
};

enum class candidate_origin : u32 {
    backend_native = 0u,
    generic_epilogue_composed = 1u
};

struct planner_candidate {
    SpMMBackend *backend = nullptr;
    backend_capability capability{};
    candidate_origin origin = candidate_origin::backend_native;
};

enum class cache_lookup_state : u32 {
    not_configured = 0u,
    miss = 1u,
    hit = 2u,
    stale_or_illegal = 3u
};

using plan_cache_lookup_fn = bool (*)(
    void *context,
    const operation_signature &operation,
    u64 device_fingerprint,
    u64 toolchain_fingerprint,
    execution_plan *out) noexcept;

struct plan_cache_lookup_hook {
    void *context = nullptr;
    plan_cache_lookup_fn lookup = nullptr;
};

// Zero limits mean that the corresponding physical/default bound applies.
// This policy only performs inexpensive structural pruning; measured selection
// and persistence remain the autotuner/cache owner's responsibility.
struct planner_policy {
    u64 workspace_soft_limit_bytes = 0u;
    std::size_t candidate_limit = max_planner_candidate_count;
};

struct planner_input {
    const math_request *request = nullptr;
    const DeviceCapabilities *device = nullptr;
    const DeviceFingerprint *fingerprint = nullptr;
    const SpMMBackendRegistry *registry = nullptr;
    plan_cache_lookup_hook cache{};
    planner_policy policy{};
};

struct planner_diagnostics {
    std::size_t registered_backend_count = 0u;
    std::size_t queried_backend_count = 0u;
    std::size_t legal_candidate_count = 0u;
    std::size_t composed_epilogue_count = 0u;
    std::size_t structurally_pruned_count = 0u;
    std::size_t malformed_candidate_count = 0u;
    std::size_t rejection_count[10]{};
    backend_capability first_rejection{};
    bool has_rejection = false;
};

// Planner results are transient control-plane state. Only `plan` is the
// immutable pointer-free decision record suitable for cache persistence.
struct planner_result {
    u32 schema_version = math_planner_schema_version;
    trivial_operation_kind trivial = trivial_operation_kind::none;
    planner_candidate candidates[max_planner_candidate_count]{};
    std::size_t candidate_count = 0u;
    std::size_t selected_index = max_planner_candidate_count;
    SpMMBackend *selected_backend = nullptr;
    execution_plan plan{};
    cache_lookup_state cache_state = cache_lookup_state::not_configured;
    planner_diagnostics diagnostics{};
};

planner_status plan_spmm(
    const planner_input &input,
    planner_result *out) noexcept;

} // namespace cellerator::compute::math
