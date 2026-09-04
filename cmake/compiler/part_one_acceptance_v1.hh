#pragma once

#include <array>
#include <string_view>

namespace cellerator::compiler::acceptance::v1 {

inline constexpr std::array<std::string_view, 12> component_registry = {
    "source", "frontend", "sema", "profiles", "semantic-ir", "planning-ir",
    "realization-ir", "reflection", "passes", "lto", "tooling", "sdk"};

struct migration_boundary {
    std::string_view compiler_owner;
    std::string_view storage_runtime_owner;
    std::string_view source_repository;
    std::string_view source_revision;
    bool compatibility_adapter;
};

inline constexpr migration_boundary jbc_migration = {
    "Cellerator", "CellShard", "tumlinso/CellShard",
    "b9749ad3e5146a04f847533d8c6f1a54146aed20", true};

inline constexpr std::array<std::string_view, 8> language_conformance = {
    "file-local pragma", "nested fields", "profile required", "explicit effects",
    "control hierarchy", "ordinary C++ preserved", "structured diagnostics",
    "implementation-defined target costs"};

inline constexpr std::array<std::string_view, 11> ir_conformance = {
    "semantic-ir", "planning-ir", "realization-ir", "reflection", "inline-ir",
    "custom-passes", "staging", "extensions", "trust-modes", "native-boundary", "lto"};

inline constexpr std::array<std::string_view, 10> guide_examples = {
    "minimal", "profiles", "planning", "realization", "custom-pass",
    "unsafe-native", "lto", "sdk", "celleratord", "ordinary-cxx"};

inline constexpr std::array<std::string_view, 8> architecture_records = {
    "directory-layout", "ownership", "jbc-provenance", "superseded-charters",
    "interfaces", "build-modes", "backends", "part-two-seam"};

inline constexpr std::array<std::string_view, 8> host_sdk_artifacts = {
    "cellerator", "libCellerator", "celleratord", "stdlib", "profiles",
    "cmake-package", "ordinary-cxx", "ceir"};

inline constexpr std::array<std::string_view, 9> nvidia_acceptance = {
    "profile-relation", "generated-candidate", "prelinked-candidate", "inline-ir",
    "custom-pass", "graph-readiness", "direct-ptx-experiment", "mixed-lto", "exact-output"};

inline constexpr std::size_t nvidia_sm = 70;
inline constexpr std::size_t nvidia_complete_cost_components = 6;

inline constexpr std::array<std::string_view, 20> final_capabilities = {
    "driver", "pragma-parser", "profiles", "semantic-ir", "planning-ir",
    "realization-ir", "ceir-round-trip", "ceir-input", "reflection", "passes",
    "self-transform", "unsafe-native", "cpu-object", "nvidia-object",
    "toolchain-overrides", "jbc-migration", "lto", "sdk-stdlib", "celleratord",
    "provenance"};

struct deferred_seam {
    std::string_view name;
    std::string_view retained_interface;
    bool part_one_prerequisite;
};

inline constexpr std::array<deferred_seam, 2> deferred_part_two = {{
    {"general-jit", "AOT object and writable CEIR contracts", false},
    {"deep-cellshard-runtime", "versioned opaque materialization request", false},
}};

struct performance_review_item {
    std::string_view subject;
    std::string_view baseline;
    std::string_view identity;
    std::string_view disposition;
};

inline constexpr std::array<performance_review_item, 6> performance_review = {{
    {"runtime-provider", "CE-GEO V100 strongest legal provider", "Tesla V100 sm_70", "retained"},
    {"compiler-overhead", "bounded compiler gates", "CCP1 build identity", "accepted"},
    {"generated-execution", "native and compatibility matrix", "CE-GEO evidence", "retained"},
    {"planning-quality", "complete-cost exact coverage", "Planning IR v1", "accepted"},
    {"object-size", "AOT compiler object baseline", "Part One toolchain manifest", "accepted"},
    {"editor-latency", "10 ms semantic background budget", "celleratord I40 v1", "accepted"},
}};

}  // namespace cellerator::compiler::acceptance::v1
