#pragma once

#include <cstdint>

namespace cellerator::compiler::sema::v1 {

struct generation_state {
    std::uint64_t structure_epoch = 0;
    std::uint64_t value_generation = 0;
    std::uint64_t active_support_generation = 0;
    std::uint64_t order_generation = 0;
};

enum class publication_state : std::uint8_t { unpublished = 1, staged, published };
enum class generation_validation : std::uint8_t {
    ok = 0,
    stale_structure,
    stale_values,
    stale_active_support,
    stale_order,
    unpublished
};

struct generation_requirement {
    generation_state expected{};
    publication_state required_publication = publication_state::published;
};

struct expert_generation_override {
    bool explicitly_unsafe = false;
    generation_validation permitted_mismatch = generation_validation::ok;
};

generation_validation validate_generations(
    const generation_requirement &requirement,
    const generation_state &actual,
    publication_state publication) noexcept;
bool generation_override_allows(generation_validation failure,
                                const expert_generation_override &override) noexcept;

}  // namespace cellerator::compiler::sema::v1
