#include <Cellerator/modeling/cuda/quantization.hh>

// Compile/link fixture for the optional CUDA quantizer. It intentionally does
// not select or launch a GPU; hardware execution requires scheduler admission.
int main() {
  using namespace cellerator::modeling::cuda;
  QuantizationProfileView profile{};
  PackedStateView packed{};
  ConstPackedStateView const_packed{};
  OverrideView overrides{};
  ConstOverrideView const_overrides{};
  DeviceStatus status{};

  (void)&reset_status;
  (void)&reset_override_count;
  (void)&validate_profile;
  (void)&encode_state_fp32;
  (void)&encode_state_fp16;
  (void)&decode_state_fp32;
  (void)&decode_state_fp16;
  (void)&update_codes_fp32;
  (void)&update_codes_fp16;

  return profile.state_feature_count + static_cast<std::uint32_t>(packed.word_capacity) +
         static_cast<std::uint32_t>(const_packed.word_capacity) +
         overrides.capacity + const_overrides.capacity +
         static_cast<std::uint32_t>(status.bits != nullptr);
}
