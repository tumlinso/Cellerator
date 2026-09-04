#pragma once

#include <Cellerator/compiler/discovery/import_the_common_jbc_atom_identity_adapters_v1.hh>

#include <cstdint>
#include <vector>

namespace Cellerator::compiler::discovery {

struct support_relation_view_v1 {
    persistent_atom_identity_v1 relation_identity{};
    const std::uint64_t* destination_offsets = nullptr;
    const std::uint64_t* source_identities = nullptr;
    std::uint32_t destination_count = 0;
    std::uint64_t edge_count = 0;
};

struct support_signature_config_v1 {
    std::uint32_t sketch_size = 0;
    std::uint32_t top_l = 0;
    std::uint64_t seed_namespace = 0;
    persistent_atom_identity_v1 biological_stratum{};
};

struct support_signature_proposal_v1 {
    std::uint32_t first_destination = 0;
    std::uint32_t second_destination = 0;
    std::uint32_t matching_minima = 0;
    std::uint32_t sketch_size = 0;
    std::uint64_t first_degree = 0;
    std::uint64_t second_degree = 0;
    persistent_atom_identity_v1 biological_stratum{};
};

struct support_signature_discovery_v1 {
    std::vector<std::uint64_t> minima;
    std::vector<support_signature_proposal_v1> proposals;
    std::uint64_t hashed_edges = 0;
    std::uint64_t compared_pairs = 0;
};

enum class support_signature_status_v1 : std::uint8_t {
    success = 0,
    invalid_relation,
    invalid_offsets,
    invalid_config,
};

[[nodiscard]] support_signature_status_v1 discover_support_signatures_v1(
    support_relation_view_v1 relation,
    support_signature_config_v1 config,
    support_signature_discovery_v1* output) noexcept;

}  // namespace Cellerator::compiler::discovery
