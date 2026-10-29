#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace cellerator::interop::ggml {

// One contiguous row-major quantized payload. Row width and original shape
// remain explicit because ggml quantization is block- and row-dependent.
struct quantized_values {
    std::string format;
    std::vector<std::int64_t> shape;
    std::vector<std::uint8_t> payload;
};

// Supported formats are selected by name (e.g. "q1_0", "q4_0", "q8_0").
// Formats requiring an importance matrix are intentionally excluded from this
// simple API rather than silently quantized with invented importance weights.
[[nodiscard]] std::vector<std::string> supported_formats();

[[nodiscard]] quantized_values quantize(
    std::span<const float> values,
    std::span<const std::int64_t> shape,
    std::string_view format);

[[nodiscard]] std::vector<float> dequantize(const quantized_values& values);

} // namespace cellerator::interop::ggml
