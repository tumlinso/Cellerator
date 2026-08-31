#pragma once

#include <Cellerator/compute/training/v2/generation_readiness.hh>
#include <Cellerator/compute/training/v2/relation_closure.hh>
#include <Cellerator/compute/training/v2/value_modes.hh>
#include <Cellerator/execution/training_program_v2/interface.hh>

#include <cstdint>

namespace cellerator::compute::training_v2 {

inline constexpr std::uint32_t frozen_training_compute_interface_version_v2 =
    2u;
inline constexpr const char frozen_training_compute_interface_name_v2[] =
    "cellerator-training-compute-v2";

} // namespace cellerator::compute::training_v2
