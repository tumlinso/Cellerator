#pragma once

#include <Cellerator/memory/cuda/resources.hh>
#include <Cellerator/tensor/state_codes/operations.hh>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace cellerator::tensor::state_codes {

// Owns one immutable feature-indexed codebook on a single CUDA device.
// Profile identity and biological feature binding remain with the caller.
class Profile final {
 public:
  Profile(std::vector<std::uint32_t> offsets,
          std::vector<float> levels,
          std::vector<std::uint32_t> feature_map,
          std::uint32_t code_bits, const memory::cuda::Stream& stream);
  Profile(const Profile&) = delete;
  Profile& operator=(const Profile&) = delete;
  Profile(Profile&&) = delete;
  Profile& operator=(Profile&&) = delete;

  // Borrowed pointers; this Profile retains their device allocations.
  ProfileView view() const & noexcept;
  ProfileView view() const && = delete;
  int device() const noexcept { return device_; }
  std::uint32_t feature_count() const noexcept { return state_features_; }
  std::uint32_t bits() const noexcept { return code_bits_; }

 private:
  int device_ = 0;
  std::uint32_t profile_features_ = 0;
  std::uint32_t state_features_ = 0;
  std::uint32_t level_count_ = 0;
  std::uint32_t code_bits_ = 0;
  memory::cuda::DeviceBuffer offsets_;
  memory::cuda::DeviceBuffer levels_;
  memory::cuda::DeviceBuffer feature_map_;
};

// A device-local, mutable quantized state value. Packed low-bit codes and
// exact FP32 overrides are separate physical components; this is not a sparse
// matrix format. It retains the immutable profile owner and uses caller-owned
// streams for every operation. Cross-stream dependencies require caller
// events or synchronization.
class NativeTensor final {
 public:
  NativeTensor(std::shared_ptr<const Profile> profile, std::uint32_t row_count,
               std::uint32_t exact_override_capacity = 0);

  // Data operations enqueue on the provided stream and return asynchronously.
  void encode(memory::cuda::ConstDeviceView input, memory::cuda::DType dtype,
              memory::cuda::ConstDeviceView exact_mask,
              const memory::cuda::Stream& stream);
  void decode(memory::cuda::DeviceView output, memory::cuda::DType dtype,
              const memory::cuda::Stream& stream) const;

  // Returns borrowed pointer views. Retain this owner until all queued users
  // have completed; the distributed execution layer can retain the owner in
  // its read lease while these views are in flight.
  MutableView view() & noexcept;
  ConstView view() const & noexcept;
  MutableView view() && = delete;
  ConstView view() const && = delete;

  // These inspections copy back a device scalar and synchronize the stream.
  std::uint32_t status(const memory::cuda::Stream& stream) const;
  std::uint32_t exact_override_count(const memory::cuda::Stream& stream) const;
  std::uint32_t rows() const noexcept { return rows_; }
  std::uint32_t features() const noexcept { return profile_->feature_count(); }
  std::size_t packed_bytes() const noexcept { return packed_.bytes(); }
  std::uint32_t exact_override_capacity() const noexcept { return capacity_; }
  int device() const noexcept { return profile_->device(); }
  bool encoded() const noexcept { return encoded_; }
  std::shared_ptr<const Profile> profile() const noexcept { return profile_; }

 private:
  ExactOverrideView mutable_overrides() noexcept;
  ConstExactOverrideView const_overrides() const noexcept;

  std::shared_ptr<const Profile> profile_;
  std::uint32_t rows_ = 0;
  std::uint32_t capacity_ = 0;
  memory::cuda::DeviceBuffer packed_;
  memory::cuda::DeviceBuffer indices_;
  memory::cuda::DeviceBuffer values_;
  memory::cuda::DeviceBuffer count_;
  mutable memory::cuda::DeviceBuffer status_;
  bool encoded_ = false;
};

}  // namespace cellerator::tensor::state_codes
