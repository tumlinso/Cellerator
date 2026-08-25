#pragma once

#include <Cellerator/compute/math/operation_core/operation_core.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::planner {

namespace operation_core = compute::math::core;

inline constexpr std::uint32_t planner_schema_version = 2u;
inline constexpr std::uint32_t objective_v2_schema_version = 2u;
inline constexpr std::uint32_t maximum_planner_candidates = 32u;
inline constexpr std::uint32_t connected_planner_schema_version = 1u;
inline constexpr std::uint32_t maximum_connected_operations = 8u;
inline constexpr std::uint32_t maximum_connected_stage_candidates = 8u;
inline constexpr std::uint32_t maximum_connected_transitions = 128u;

struct mathematical_problem_key {
    operation_core::stable_id identity{};
};

struct semantic_geometry_key {
    execution::domain_id source_domain{};
    execution::domain_id destination_domain{};
    execution::geometry_id geometry{};
    execution::order_id source_order{};
    execution::order_id destination_order{};
    execution::partition_id partition{};
};

struct device_performance_key {
    std::uint32_t vendor = 0u;
    std::uint16_t architecture_major = 0u;
    std::uint16_t architecture_minor = 0u;
    std::uint64_t performance_class = 0u;
};

struct runtime_build_key {
    std::uint64_t runtime = 0u;
    std::uint64_t kernel_build = 0u;
    std::uint64_t driver = 0u;
    std::uint64_t library = 0u;
};

struct persistent_structure_dependency {
    execution::structure_id identity{};
    execution::structure_epoch epoch{};
};

struct persistent_structure_set_key {
    persistent_structure_dependency
        structures[execution::maximum_operation_structures]{};
    std::uint32_t count = 0u;
    std::uint32_t reserved = 0u;
};

struct persistent_projection_key {
    execution::projection_id identity{};
    operation_core::projection_kind kind =
        operation_core::projection_kind::native_row_masked;
    std::uint16_t schema_version = 0u;
    std::uint32_t variant = 0u;
};

struct policy_reuse_key {
    std::uint64_t structure_reuse = 1u;
    std::uint64_t projection_reuse = 1u;
    std::uint64_t value_reuse = 1u;
    std::uint32_t numeric_policy = 0u;
    std::uint32_t determinism_policy = 0u;
    std::uint32_t output_order_policy = 0u;
    std::uint32_t graph_policy = 0u;
};

// Cache identity stays factored. Live pointers and mutable value generations
// are launch validation state and deliberately have no field here.
struct planning_keys {
    mathematical_problem_key problem{};
    persistent_structure_set_key structures{};
    semantic_geometry_key geometry{};
    device_performance_key device{};
    runtime_build_key build{};
    policy_reuse_key policy{};
};

struct phase_costs {
    double host_preparation_ns = 0.0;
    double semantic_packing_ns = 0.0;
    double projection_construction_ns = 0.0;
    double backend_prepare_ns = 0.0;
    double static_value_pack_ns = 0.0;
    double h2d_ns = 0.0;
    double dynamic_input_pack_ns = 0.0;
    double kernel_ns = 0.0;
    double epilogue_ns = 0.0;
    double order_transform_ns = 0.0;
    double synchronization_ns = 0.0;
    double communication_ns = 0.0;
    double d2h_ns = 0.0;
    std::uint64_t h2d_bytes = 0u;
    std::uint64_t communication_bytes = 0u;
    std::uint64_t d2h_bytes = 0u;
    std::uint64_t persistent_bytes = 0u;
    std::uint64_t transient_bytes = 0u;
};

struct total_cost {
    phase_costs phases{};
    std::uint64_t structure_reuse = 1u;
    std::uint64_t projection_reuse = 1u;
    std::uint64_t value_reuse = 1u;
    double amortized_total_ns = 0.0;
};

enum candidate_policy_flag : std::uint32_t {
    planner_candidate_correct = 1u << 0u,
    planner_candidate_deterministic = 1u << 1u,
    planner_candidate_graph_capture = 1u << 2u,
    planner_candidate_conventional = 1u << 3u,
    planner_candidate_empirical_required = 1u << 4u
};

struct planner_candidate {
    operation_core::stable_id identity{};
    const char *name = nullptr;
    const operation_core::operation_candidate *operation = nullptr;
    operation_core::projection_key projection{};
    phase_costs analytical{};
    std::uint32_t flags = 0u;
    std::uint32_t reserved = 0u;
};

struct measured_candidate {
    bool correct = false;
    bool contaminated = false;
    std::uint16_t reserved = 0u;
    std::uint32_t sample_count = 0u;
    phase_costs phases{};
    double spread_percent = 0.0;
};

using measurement_function = bool (*)(
    void *context,
    const planner_candidate &candidate,
    measured_candidate *measurement) noexcept;

struct measurement_hook {
    void *context = nullptr;
    measurement_function measure = nullptr;
};

struct empirical_evidence {
    std::uint64_t evidence_revision = 0u;
    std::uint32_t sample_count = 0u;
    std::uint32_t reserved = 0u;
    double median_total_ns = 0.0;
    double spread_percent = 0.0;
    double confidence = 0.0;
    double practical_tolerance_percent = 0.0;
};

struct plan_cache_entry {
    planning_keys keys{};
    operation_core::stable_id winner{};
    persistent_projection_key winner_projection{};
    empirical_evidence evidence{};
    bool occupied = false;
};

using cache_lookup_function = bool (*)(
    void *context,
    const planning_keys &keys,
    plan_cache_entry *entry) noexcept;
using cache_store_function = bool (*)(
    void *context,
    const plan_cache_entry &entry) noexcept;

struct plan_cache_hooks {
    void *context = nullptr;
    cache_lookup_function lookup = nullptr;
    cache_store_function store = nullptr;
};

struct planner_policy {
    std::uint32_t shortlist_size = 3u;
    std::uint32_t maximum_measurements = 3u;
    std::uint64_t minimum_tuning_work_items = 4096u;
    std::uint64_t maximum_persistent_bytes = 0u;
    std::uint64_t maximum_transient_bytes = 0u;
    double practical_tolerance_percent = 2.0;
    double maximum_spread_percent = 10.0;
    double minimum_cache_confidence = 0.8;
    bool deterministic = false;
    bool graph_capture_required = false;
    bool tune_one_shot = false;
    bool allow_analytical_fallback_after_measurement_failure = true;
};

struct planner_request {
    std::uint32_t schema_version = planner_schema_version;
    operation_core::operation_problem problem{};
    planning_keys keys{};
    const planner_candidate *candidates = nullptr;
    std::uint32_t candidate_count = 0u;
    std::uint32_t reserved = 0u;
    planner_policy policy{};
    measurement_hook measurement{};
    plan_cache_hooks cache{};
    std::uint64_t current_evidence_revision = 0u;
};

enum class candidate_rejection : std::uint8_t {
    none = 0u,
    malformed = 1u,
    incorrect = 2u,
    nondeterministic = 3u,
    graph_incompatible = 4u,
    persistent_memory = 5u,
    transient_memory = 6u,
    measurement_failed = 7u,
    contaminated = 8u
};

enum class selection_source : std::uint8_t {
    analytical = 1u,
    empirical = 2u,
    cache = 3u
};

enum class cache_state : std::uint8_t {
    not_configured = 0u,
    miss = 1u,
    hit = 2u,
    stale = 3u,
    winner_unavailable = 4u
};

enum class planner_status_code : std::uint8_t {
    ok = 0u,
    invalid_argument = 1u,
    invalid_cost = 2u,
    no_legal_candidate = 3u,
    no_correct_measurement = 4u
};

struct planner_status {
    planner_status_code code = planner_status_code::ok;
    const char *message = "ok";

    constexpr explicit operator bool() const noexcept {
        return code == planner_status_code::ok;
    }
};

struct candidate_diagnostic {
    operation_core::stable_id identity{};
    candidate_rejection rejection = candidate_rejection::none;
    bool shortlisted = false;
    bool measured = false;
    bool conventional = false;
    std::uint8_t reserved[4]{};
    std::uint32_t sample_count = 0u;
    std::uint32_t reserved_count = 0u;
    double spread_percent = 0.0;
    total_cost analytical{};
    total_cost empirical{};
};

struct planner_result {
    std::uint32_t schema_version = planner_schema_version;
    planner_status status{};
    operation_core::stable_id winner{};
    const planner_candidate *selected = nullptr;
    selection_source source = selection_source::analytical;
    cache_state cache = cache_state::not_configured;
    bool tuning_skipped = false;
    bool conventional_winner = false;
    bool cache_store_failed = false;
    std::uint8_t reserved{};
    std::uint32_t legal_count = 0u;
    std::uint32_t shortlist_count = 0u;
    std::uint32_t measurement_count = 0u;
    double confidence = 0.0;
    double practical_tolerance_percent = 0.0;
    const char *reason = nullptr;
    candidate_diagnostic diagnostics[maximum_planner_candidates]{};
};

planner_status compute_total_cost(
    const phase_costs &phases,
    std::uint64_t structure_reuse,
    std::uint64_t projection_reuse,
    std::uint64_t value_reuse,
    total_cost *out) noexcept;
planner_status plan_end_to_end(
    const planner_request &request,
    planner_result *out) noexcept;
bool same_planning_keys(
    const planning_keys &lhs,
    const planning_keys &rhs) noexcept;
bool make_persistent_structure_set_key(
    const operation_core::structure_set_key &live,
    persistent_structure_set_key *persistent) noexcept;
bool same_persistent_projection_key(
    const persistent_projection_key &persistent,
    const operation_core::projection_key &live) noexcept;

// Bounded connected-operation planner. It deliberately models a linear chain:
// later DAG/general-graph work may replace this interface without changing the
// single-operation planner or candidate contracts.
struct connected_operation_stage {
    operation_core::operation_problem problem{};
    planning_keys keys{};
    planner_policy policy{};
    const planner_candidate *candidates = nullptr;
    std::uint32_t candidate_count = 0u;
    std::uint32_t reserved = 0u;
};

struct connected_transition_cost {
    std::uint32_t boundary = 0u;
    operation_core::stable_id producer{};
    operation_core::stable_id consumer{};
    execution::order_transition_kind order =
        execution::order_transition_kind::preserve;
    bool format_conversion = false;
    bool legal = true;
    std::uint16_t reserved = 0u;
    operation_core::stable_id conversion{};
    phase_costs phases{};
};

struct connected_plan_path {
    std::uint32_t stage_count = 0u;
    std::uint32_t reserved = 0u;
    operation_core::stable_id candidates[maximum_connected_operations]{};
    persistent_projection_key projections[maximum_connected_operations]{};
};

struct connected_planning_keys {
    operation_core::stable_id graph_identity{};
    std::uint32_t stage_count = 0u;
    std::uint32_t reserved = 0u;
    planning_keys stages[maximum_connected_operations]{};
};

struct measured_connected_plan {
    bool correct = false;
    bool contaminated = false;
    std::uint16_t reserved = 0u;
    std::uint32_t sample_count = 0u;
    double amortized_total_ns = 0.0;
    double spread_percent = 0.0;
};

using connected_measurement_function = bool (*)(
    void *context,
    const connected_plan_path &path,
    measured_connected_plan *measurement) noexcept;

struct connected_measurement_hook {
    void *context = nullptr;
    connected_measurement_function measure = nullptr;
};

struct connected_plan_cache_entry {
    connected_planning_keys keys{};
    connected_plan_path winner{};
    empirical_evidence evidence{};
    bool occupied = false;
};

using connected_cache_lookup_function = bool (*)(
    void *context,
    const connected_planning_keys &keys,
    connected_plan_cache_entry *entry) noexcept;
using connected_cache_store_function = bool (*)(
    void *context,
    const connected_plan_cache_entry &entry) noexcept;

struct connected_plan_cache_hooks {
    void *context = nullptr;
    connected_cache_lookup_function lookup = nullptr;
    connected_cache_store_function store = nullptr;
};

struct connected_planner_request {
    std::uint32_t schema_version = connected_planner_schema_version;
    operation_core::stable_id graph_identity{};
    const connected_operation_stage *stages = nullptr;
    std::uint32_t stage_count = 0u;
    std::uint32_t reserved = 0u;
    const connected_transition_cost *transitions = nullptr;
    std::uint32_t transition_count = 0u;
    std::uint32_t shortlist_size = 3u;
    std::uint32_t maximum_measurements = 3u;
    double practical_tolerance_percent = 2.0;
    double maximum_spread_percent = 10.0;
    double minimum_cache_confidence = 0.8;
    bool force_empirical = false;
    bool allow_analytical_fallback_after_measurement_failure = true;
    std::uint8_t reserved_flags[6]{};
    connected_measurement_hook measurement{};
    connected_plan_cache_hooks cache{};
    std::uint64_t current_evidence_revision = 0u;
};

struct connected_stage_selection {
    const planner_candidate *candidate = nullptr;
    total_cost analytical{};
};

struct connected_planner_result {
    std::uint32_t schema_version = connected_planner_schema_version;
    planner_status status{};
    connected_plan_path winner{};
    connected_stage_selection stages[maximum_connected_operations]{};
    selection_source source = selection_source::analytical;
    cache_state cache = cache_state::not_configured;
    bool cache_store_failed = false;
    bool empirical_required = false;
    std::uint16_t reserved = 0u;
    std::uint32_t legal_path_count = 0u;
    std::uint32_t shortlist_count = 0u;
    std::uint32_t measurement_count = 0u;
    double analytical_total_ns = 0.0;
    double empirical_total_ns = 0.0;
    double confidence = 0.0;
    const char *reason = nullptr;
};

bool same_connected_planning_keys(
    const connected_planning_keys &lhs,
    const connected_planning_keys &rhs) noexcept;

planner_status plan_connected_operations(
    const connected_planner_request &request,
    connected_planner_result *out) noexcept;

// Versioned CP-BP objective v2. This is operation-aware planner input and does
// not alter packing_exact_objective_kind or any CPK1 v1 bytes.
struct objective_v2_statistics {
    std::uint64_t useful_edges = 0u;
    std::uint64_t metadata_bytes = 0u;
    std::uint64_t value_bytes = 0u;
    std::uint64_t partial_block_slots = 0u;
    std::uint64_t cross_partition_edges = 0u;
    double feature_reuse = 0.0;
    double row_imbalance = 0.0;
    double module_overlap = 0.0;
    double module_activation_frequency = 0.0;
    double transpose_locality = 0.0;
    double quantization_outlier_fraction = 0.0;
};

struct objective_v2_context {
    operation_core::operation_kind operation =
        operation_core::operation_kind::sparse_dense_multiply;
    std::uint32_t dense_width = 1u;
    execution::numeric_type value_type = execution::numeric_type::f32;
    std::uint32_t registers_per_thread = 0u;
    std::uint32_t shared_bytes_per_block = 0u;
    std::uint64_t expected_reuse = 1u;
    bool transpose_required = false;
    bool epilogue_fused = false;
    bool canonical_output_required = false;
    bool quantized = false;
};

struct objective_v2_weights {
    double byte_cost = 1.0;
    double partial_block_cost = 1.0;
    double imbalance_cost = 1.0;
    double register_pressure_cost = 1.0;
    double shared_pressure_cost = 1.0;
    double order_transform_cost = 1.0;
    double transpose_cost = 1.0;
    double communication_cost = 1.0;
    double quantization_cost = 1.0;
    double reuse_credit = 1.0;
    double epilogue_credit = 1.0;
    double module_credit = 0.0;
};

struct objective_v2_result {
    std::uint32_t schema_version = objective_v2_schema_version;
    double storage = 0.0;
    double execution = 0.0;
    double order_and_transpose = 0.0;
    double communication = 0.0;
    double quantization = 0.0;
    double credits = 0.0;
    double score = 0.0;
};

planner_status evaluate_objective_v2(
    const objective_v2_statistics &statistics,
    const objective_v2_context &context,
    const objective_v2_weights &weights,
    objective_v2_result *out) noexcept;

static_assert(std::is_trivially_copyable<planning_keys>::value,
    "planner keys must remain persistable without pointers");
static_assert(std::is_trivially_copyable<persistent_projection_key>::value,
    "cached projection identity must remain persistable without handles");
static_assert(std::is_trivially_copyable<phase_costs>::value,
    "phase costs must remain evidence-record friendly");
static_assert(std::is_trivially_copyable<objective_v2_result>::value,
    "objective v2 results must remain evidence-record friendly");
static_assert(std::is_trivially_copyable<connected_plan_path>::value,
    "connected plan paths must remain pointer-free cache records");
static_assert(std::is_trivially_copyable<connected_planning_keys>::value,
    "connected planning keys must remain pointer-free cache records");

} // namespace cellerator::planner
