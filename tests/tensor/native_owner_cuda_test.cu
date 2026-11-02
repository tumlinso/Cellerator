#include <Cellerator/memory/cuda/resources.hh>
#include <Cellerator/memory/cuda/shared_buffer.hh>
#include <Cellerator/tensor/state_codes/owner.hh>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace memory_cuda = ::cellerator::memory::cuda;
namespace state_codes = ::cellerator::tensor::state_codes;

namespace {

static_assert(!std::is_copy_constructible_v<state_codes::Profile>);
static_assert(!std::is_move_constructible_v<state_codes::Profile>);

constexpr std::uint32_t kRows = 3;
constexpr std::uint32_t kFeatures = 4;
constexpr std::size_t kElementCount = kRows * kFeatures;

const std::array<float, kElementCount> kState{
    4.0f, 1.0f, 6.25f, 2.0f,
    5.0f, 0.0f, 7.0f, 3.0f,
    4.0f, 0.0f, 7.0f, 3.0f,
};

std::vector<std::uint8_t> exact_mask() {
  std::vector<std::uint8_t> mask(kElementCount, 0U);
  mask[2] = 1U;
  return mask;
}

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void qualify_shared_buffer(int device, const memory_cuda::Stream& stream) {
  constexpr std::size_t count = 7;
  const std::array<float, count> source{0.25f, 1.5f, -2.0f, 4.25f,
                                        8.0f, -0.5f, 3.0f};
  memory_cuda::SharedDeviceBuffer<float> retained;
  {
    auto original = memory_cuda::SharedDeviceBuffer<float>::allocate(count, device);
    require(original.device() == device, "shared allocation ignored its explicit device");
    require(original.size() == count && original.data() != nullptr,
            "shared allocation has the wrong extent");

    retained = original;
    require(retained.retained_owner() != nullptr,
            "copied shared handle did not retain the allocation owner");
    original.upload_async(source.data(), source.size(), stream);
    stream.synchronize();

    // Release the original handle while a copied owner remains live.
    original = memory_cuda::SharedDeviceBuffer<float>{};
    require(original.empty() && original.data() == nullptr,
            "reset shared handle still exposes its old allocation");
  }

  const auto view = retained.view();
  require(view.data != nullptr && view.count == count,
          "retained shared view lost its allocation or extent");
  require(view.where.kind == ::cellerator::memory::domain::device &&
              view.where.device_ordinal == device,
          "retained shared view reports the wrong placement");

  std::array<float, count> recovered{};
  retained.download_async(recovered.data(), recovered.size(), stream);
  stream.synchronize();
  for (std::size_t i = 0; i < count; ++i) {
    require(recovered[i] == source[i], "shared-buffer stream readback mismatch");
  }
}

std::shared_ptr<const state_codes::Profile> make_profile(
    std::uint32_t bits, int device, const memory_cuda::Stream& stream) {
  // Each profile feature has a distinct codebook. State columns deliberately
  // use a non-identity map so accidentally indexing by column cannot pass.
  std::vector<std::uint32_t> offsets{0U, 2U, 4U, 6U, 8U};
  std::vector<float> levels{0.0f, 1.0f, 2.0f, 3.0f,
                            4.0f, 5.0f, 6.0f, 7.0f};
  std::vector<std::uint32_t> feature_map{2U, 0U, 3U, 1U};
  auto profile = std::make_shared<state_codes::Profile>(
      std::move(offsets), std::move(levels), std::move(feature_map), bits, stream);
  require(profile->device() == device, "profile was not placed on requested device");
  require(profile->feature_count() == kFeatures,
          "profile feature map has the wrong state width");
  require(profile->bits() == bits, "profile reports an unexpected code width");
  return profile;
}

template <typename Input>
std::vector<float> make_expected(const std::array<float, kElementCount>& input) {
  std::vector<float> expected(kElementCount);
  for (std::size_t i = 0; i < kElementCount; ++i) {
    if constexpr (std::is_same_v<Input, __half>) {
      expected[i] = __half2float(__float2half_rn(input[i]));
    } else {
      expected[i] = input[i];
    }
  }
  return expected;
}

template <typename Input>
void qualify_state_codes(std::uint32_t bits, int device,
                         const memory_cuda::Stream& stream,
                         const char* input_name) {
  auto host_input = make_expected<Input>(kState);
  std::vector<Input> encoded_input(kElementCount);
  for (std::size_t i = 0; i < kElementCount; ++i) {
    if constexpr (std::is_same_v<Input, __half>) {
      encoded_input[i] = __float2half_rn(kState[i]);
    } else {
      encoded_input[i] = kState[i];
    }
  }

  auto profile = make_profile(bits, device, stream);
  state_codes::NativeTensor tensor(profile, kRows,
                                   static_cast<std::uint32_t>(kElementCount));
  profile.reset();
  require(tensor.profile() != nullptr && tensor.profile()->device() == device,
          "native tensor did not retain its immutable profile owner");
  memory_cuda::DeviceBuffer input(
      memory_cuda::checked_bytes(kElementCount, sizeof(Input)), device);
  memory_cuda::DeviceBuffer mask(kElementCount * sizeof(std::uint8_t), device);
  memory_cuda::DeviceBuffer output(
      memory_cuda::checked_bytes(kElementCount, sizeof(float)), device);

  const auto mask_host = exact_mask();
  input.upload_async(encoded_input.data(), input.bytes(), stream);
  mask.upload_async(mask_host.data(), mask.bytes(), stream);
  stream.synchronize();

  const auto dtype = std::is_same_v<Input, __half>
                         ? memory_cuda::DType::float16
                         : memory_cuda::DType::float32;
  tensor.encode(input.view(), dtype, mask.view(), stream);
  require(tensor.encoded(), "encode did not publish the encoded state");
  require(tensor.packed_bytes() != 0U, "encode produced no packed code storage");
  require(tensor.exact_override_count(stream) == 1U,
          "exact-value sidecar did not retain exactly the requested element");

  tensor.decode(output.view(), memory_cuda::DType::float32, stream);
  std::array<float, kElementCount> recovered{};
  output.download_async(recovered.data(), output.bytes(), stream);
  stream.synchronize();

  require(tensor.status(stream) == 0U,
          "state-code operation reported profile, override, or value errors");
  for (std::size_t i = 0; i < kElementCount; ++i) {
    if (std::fabs(recovered[i] - host_input[i]) > 1.0e-6f) {
      throw std::runtime_error(std::string("state-code roundtrip mismatch for ") +
                               input_name + ", " + std::to_string(bits) +
                               " bits, element " + std::to_string(i));
    }
  }
}

void qualify_device(int device) {
  memory_cuda::DeviceGuard device_guard(device);
  memory_cuda::Stream stream(device);
  qualify_shared_buffer(device, stream);

  for (const std::uint32_t bits : {1U, 2U, 4U, 8U}) {
    qualify_state_codes<float>(bits, device, stream, "FP32 input");
    qualify_state_codes<__half>(bits, device, stream, "FP16 input");
  }
}

}  // namespace

int main(int argc, char** argv) {
  bool host_only = false;
  int device = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string argument(argv[i]);
    if (argument == "--host-only") {
      host_only = true;
    } else if (argument == "--device" && i + 1 < argc) {
      char* end = nullptr;
      const long parsed = std::strtol(argv[++i], &end, 10);
      if (end == argv[i] || *end != '\0' || parsed < 0 || parsed > 32767) {
        std::fprintf(stderr, "invalid --device ordinal\n");
        return 2;
      }
      device = static_cast<int>(parsed);
    } else {
      std::fprintf(stderr,
                   "usage: %s [--device ordinal] [--host-only]\n", argv[0]);
      return 2;
    }
  }

  if (host_only) {
    // This branch intentionally avoids every CUDA runtime/API call. It is a
    // safe smoke check when compilation is permitted but device execution is not.
    std::puts("native owner host-only contract check passed; no CUDA calls made");
    return 0;
  }

  try {
    int device_count = 0;
    memory_cuda::check(cudaGetDeviceCount(&device_count), "query CUDA device count");
    require(device < device_count, "requested CUDA device ordinal is unavailable");
    qualify_device(device);
    std::printf("native memory/state-code owner qualification passed on device %d\n",
                device);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "native owner qualification failed: %s\n", error.what());
    return 1;
  }
}
