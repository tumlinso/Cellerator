#pragma once

#include <Cellerator/execution/training_program_v2/graph_capture.hh>
#include <Cellerator/execution/training_program_v2/program.hh>

#include <cstdint>

namespace cellerator::execution::training_v2 {

inline constexpr std::uint32_t frozen_training_program_interface_version_v2 =
    2u;
inline constexpr const char frozen_training_program_interface_name_v2[] =
    "cellerator-training-program-v2";

} // namespace cellerator::execution::training_v2
