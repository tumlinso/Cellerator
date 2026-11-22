#include <Cellerator/tensor/state_codes/owner.hh>

#include <limits>
#include <stdexcept>
#include <utility>

namespace cellerator::tensor::state_codes {
namespace {

std::size_t element_count(std::uint32_t rows, std::uint32_t features) {
  return memory::cuda::checked_bytes(static_cast<std::size_t>(rows),
                                     static_cast<std::size_t>(features));
}

void check_stream(int device, const memory::cuda::Stream& stream) {
  if (device != stream.device()) {
    throw std::invalid_argument("state-code tensor/stream device mismatch");
  }
}

void aligned(memory::cuda::ConstDeviceView view, std::size_t alignment) {
  if (reinterpret_cast<std::uintptr_t>(view.data) % alignment != 0U) {
    throw std::invalid_argument("device view is not aligned for its numeric type");
  }
}

void aligned(memory::cuda::DeviceView view, std::size_t alignment) {
  aligned(static_cast<memory::cuda::ConstDeviceView>(view), alignment);
}

}  // namespace

Profile::Profile(memory::const_array_view<std::uint32_t> offsets,
                 memory::const_array_view<float> levels,
                 memory::const_array_view<std::uint32_t> feature_map,
                 std::uint32_t code_bits,
                 const memory::cuda::Stream& stream)
    : device_(stream.device()), code_bits_(code_bits) {
  if (code_bits != 1U && code_bits != 2U && code_bits != 4U && code_bits != 8U) {
    throw std::invalid_argument("code bits must be 1, 2, 4 or 8");
  }
  if (!offsets.data || !levels.data || !feature_map.data ||
      offsets.where.kind == memory::domain::device ||
      levels.where.kind == memory::domain::device ||
      feature_map.where.kind == memory::domain::device ||
      offsets.count < 2U || feature_map.count == 0U ||
      offsets.count - 1U > UINT32_MAX || feature_map.count > UINT32_MAX ||
      levels.count > UINT32_MAX || offsets.data[0] != 0U ||
      offsets.data[offsets.count - 1U] != levels.count) {
    throw std::invalid_argument("invalid codebook dimensions");
  }

  profile_features_ = static_cast<std::uint32_t>(offsets.count - 1U);
  state_features_ = static_cast<std::uint32_t>(feature_map.count);
  level_count_ = static_cast<std::uint32_t>(levels.count);
  for (std::size_t profile = 0; profile < profile_features_; ++profile) {
    const auto begin = offsets.data[profile];
    const auto end = offsets.data[profile + 1U];
    if (end <= begin || end > levels.count || end - begin > (1U << code_bits)) {
      throw std::invalid_argument("codebook has invalid number of levels");
    }
    for (std::size_t level = begin; level < end; ++level) {
      if (!__builtin_isfinite(levels.data[level]) ||
          (level > begin && levels.data[level] <= levels.data[level - 1U])) {
        throw std::invalid_argument(
            "codebook levels must be finite and strictly increasing");
      }
    }
  }
  for (std::size_t feature = 0; feature < feature_map.count; ++feature) {
    if (feature_map.data[feature] >= profile_features_) {
      throw std::invalid_argument("invalid feature map");
    }
  }

  const auto offset_bytes = memory::cuda::checked_bytes(offsets.count, sizeof(*offsets.data));
  const auto level_bytes = memory::cuda::checked_bytes(levels.count, sizeof(*levels.data));
  const auto map_bytes = memory::cuda::checked_bytes(feature_map.count, sizeof(*feature_map.data));
  offsets_ = memory::cuda::DeviceBuffer(offset_bytes, device_);
  levels_ = memory::cuda::DeviceBuffer(level_bytes, device_);
  feature_map_ = memory::cuda::DeviceBuffer(map_bytes, device_);

  try {
    offsets_.upload_async(offsets.data, offset_bytes, stream);
    levels_.upload_async(levels.data, level_bytes, stream);
    feature_map_.upload_async(feature_map.data, map_bytes, stream);
    stream.synchronize();
  } catch (...) {
    // Complete queued transfers before the borrowed host arrays can retire.
    // Preserve the original error if draining also reports a CUDA failure.
    try { stream.synchronize(); } catch (...) {}
    throw;
  }
}

ProfileView Profile::view() const & noexcept {
  return {static_cast<const std::uint32_t*>(offsets_.data()),
          static_cast<const float*>(levels_.data()),
          static_cast<const std::uint32_t*>(feature_map_.data()),
          profile_features_, state_features_, level_count_, code_bits_};
}

NativeTensor::NativeTensor(std::shared_ptr<const Profile> profile,
                           std::uint32_t row_count,
                           std::uint32_t exact_override_capacity)
    : profile_(std::move(profile)), rows_(row_count),
      capacity_(exact_override_capacity) {
  if (!profile_) {
    throw std::invalid_argument("null state-code profile");
  }
  const auto count = element_count(rows_, features());
  if (count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::length_error(
        "state-code tensor exceeds 32-bit logical index capacity");
  }
  if (capacity_ > count) {
    throw std::invalid_argument("exact override capacity exceeds tensor size");
  }
  const auto bit_count =
      memory::cuda::checked_bytes(count, profile_->bits());
  const auto words = (bit_count + 31U) / 32U;
  packed_ = memory::cuda::DeviceBuffer(
      memory::cuda::checked_bytes(words, sizeof(std::uint32_t)), device());
  indices_ = memory::cuda::DeviceBuffer(
      memory::cuda::checked_bytes(capacity_, sizeof(std::uint32_t)), device());
  values_ = memory::cuda::DeviceBuffer(
      memory::cuda::checked_bytes(capacity_, sizeof(float)), device());
  count_ = memory::cuda::DeviceBuffer(sizeof(std::uint32_t), device());
  status_ = memory::cuda::DeviceBuffer(sizeof(std::uint32_t), device());
}

ExactOverrideView NativeTensor::mutable_overrides() noexcept {
  return {static_cast<std::uint32_t*>(indices_.data()),
          static_cast<float*>(values_.data()),
          static_cast<std::uint32_t*>(count_.data()), capacity_};
}

ConstExactOverrideView NativeTensor::const_overrides() const noexcept {
  if (capacity_ == 0U) {
    return {};
  }
  return {static_cast<const std::uint32_t*>(indices_.data()),
          static_cast<const float*>(values_.data()),
          static_cast<const std::uint32_t*>(count_.data()), capacity_};
}

MutableView NativeTensor::view() & noexcept {
  return {profile_->view(),
          {static_cast<std::uint32_t*>(packed_.data()),
           packed_.bytes() / sizeof(std::uint32_t)},
          mutable_overrides(),
          {static_cast<std::uint32_t*>(status_.data())},
          rows_};
}

ConstView NativeTensor::view() const & noexcept {
  return {profile_->view(),
          {static_cast<const std::uint32_t*>(packed_.data()),
           packed_.bytes() / sizeof(std::uint32_t)},
          const_overrides(),
          {static_cast<std::uint32_t*>(status_.data())},
          rows_};
}

void NativeTensor::encode(memory::cuda::ConstDeviceView input,
                          memory::cuda::DType dtype,
                          memory::cuda::ConstDeviceView exact_mask,
                          const memory::cuda::Stream& stream) {
  check_stream(device(), stream);
  const auto count = element_count(rows_, features());
  const auto value_size = memory::cuda::item_size(dtype);
  aligned(input, value_size);
  memory::cuda::require_view(
      input, memory::cuda::checked_bytes(count, value_size), device(),
      "state-code input");
  if (exact_mask.data != nullptr || exact_mask.bytes != 0U) {
    memory::cuda::require_view(exact_mask, count, device(), "exact mask");
    if (capacity_ < count) {
      throw std::invalid_argument(
          "exact-mask encoding requires full tensor override capacity");
    }
  }

  memory::cuda::DeviceGuard guard(device());
  const auto views = view();
  memory::cuda::check(reset_status(views.status, stream.get()),
                      "reset state-code status");
  memory::cuda::check(
      reset_exact_override_count(views.exact_overrides, stream.get()),
      "reset exact override count");
  memory::cuda::check(
      dtype == memory::cuda::DType::float16
          ? encode_fp16(static_cast<const __half*>(input.data), rows_,
                        views.profile, views.packed,
                        static_cast<const std::uint8_t*>(exact_mask.data),
                        views.exact_overrides, views.status, stream.get())
          : encode_fp32(static_cast<const float*>(input.data), rows_,
                        views.profile, views.packed,
                        static_cast<const std::uint8_t*>(exact_mask.data),
                        views.exact_overrides, views.status, stream.get()),
      "encode state-code tensor");
  encoded_ = true;
}

void NativeTensor::decode(memory::cuda::DeviceView output,
                          memory::cuda::DType dtype,
                          const memory::cuda::Stream& stream) const {
  if (!encoded_) {
    throw std::logic_error("state-code tensor must be encoded before decoding");
  }
  check_stream(device(), stream);
  const auto value_size = memory::cuda::item_size(dtype);
  aligned(output, value_size);
  memory::cuda::require_view(
      output, memory::cuda::checked_bytes(element_count(rows_, features()),
                                           value_size),
      device(), "state-code decode output");

  memory::cuda::DeviceGuard guard(device());
  const auto views = view();
  memory::cuda::check(
      dtype == memory::cuda::DType::float16
          ? decode_fp16(views.packed, rows_, views.profile,
                        views.exact_overrides,
                        static_cast<__half*>(output.data), views.status,
                        stream.get())
          : decode_fp32(views.packed, rows_, views.profile,
                        views.exact_overrides,
                        static_cast<float*>(output.data), views.status,
                        stream.get()),
      "decode state-code tensor");
}

std::uint32_t NativeTensor::status(
    const memory::cuda::Stream& stream) const {
  if (!encoded_) {
    throw std::logic_error("state-code tensor has not been encoded");
  }
  check_stream(device(), stream);
  memory::cuda::DeviceGuard guard(device());
  std::uint32_t result = 0;
  status_.download_async(&result, sizeof(result), stream);
  stream.synchronize();
  return result;
}

std::uint32_t NativeTensor::exact_override_count(
    const memory::cuda::Stream& stream) const {
  if (!encoded_) {
    throw std::logic_error("state-code tensor has not been encoded");
  }
  check_stream(device(), stream);
  memory::cuda::DeviceGuard guard(device());
  std::uint32_t result = 0;
  count_.download_async(&result, sizeof(result), stream);
  stream.synchronize();
  return result;
}

}  // namespace cellerator::tensor::state_codes
