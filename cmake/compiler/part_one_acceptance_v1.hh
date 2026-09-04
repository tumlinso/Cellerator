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

}  // namespace cellerator::compiler::acceptance::v1
