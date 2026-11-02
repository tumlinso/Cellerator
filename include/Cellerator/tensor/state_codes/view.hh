#pragma once

#include <cstddef>
#include <cstdint>

namespace cellerator::tensor::state_codes {

// Read-only projection of an immutable profile. Offsets delimit one sorted
// codebook per profile feature; feature_map maps each physical state column
// to a profile feature ordinal.
struct ProfileView {
  const std::uint32_t* codebook_offsets{};  // profile_feature_count + 1
  const float* levels{};                    // codebook_offsets.back()
  const std::uint32_t* feature_map{};       // state_feature_count
  std::uint32_t profile_feature_count{};
  std::uint32_t state_feature_count{};
  std::uint32_t level_count{};  // bounds device-side offset checks
  std::uint32_t code_bits{};    // one of 1, 2, 4, 8
};

struct PackedView {
  std::uint32_t* words{};
  std::size_t word_capacity{};
};

struct ConstPackedView {
  const std::uint32_t* words{};
  std::size_t word_capacity{};
};

// Exact/context-sensitive values are kept in a sparse sidecar. The kernel
// appends logical-index/value pairs through count; pair order is unspecified.
struct ExactOverrideView {
  std::uint32_t* logical_indices{};
  float* values{};
  std::uint32_t* count{};  // device scalar, cleared by reset_exact_override_count
  std::uint32_t capacity{};
};

struct ConstExactOverrideView {
  const std::uint32_t* logical_indices{};
  const float* values{};
  const std::uint32_t* count{};
  std::uint32_t capacity{};
};

// Caller-owned asynchronous validation bits. Bit 0: malformed profile (bad
// offsets, unsorted/non-finite levels, or invalid feature mapping). Bit 1:
// exact override capacity overflow. Bit 2: non-finite source/update value.
struct DeviceStatus {
  std::uint32_t* bits{};
};

// A borrowed mutable view over one device-local state-code value. Keep its
// owner alive until every operation using these pointers completes. State is
// row-major [row, feature].
struct MutableView {
  ProfileView profile{};
  PackedView packed{};
  ExactOverrideView exact_overrides{};
  DeviceStatus status{};
  std::uint32_t row_count{};
};

// Read-only code and exact-value views still carry writable status because a
// decode or fused consumer may report malformed data asynchronously.
struct ConstView {
  ProfileView profile{};
  ConstPackedView packed{};
  ConstExactOverrideView exact_overrides{};
  DeviceStatus status{};
  std::uint32_t row_count{};
};

}  // namespace cellerator::tensor::state_codes
