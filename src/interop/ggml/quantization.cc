#include <Cellerator/interop/ggml/quantization.hh>

#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

namespace cellerator::interop::ggml {
namespace {

struct format_entry {
    std::string_view name;
    ggml_type type;
};

constexpr format_entry formats[] = {
    {"q1_0", GGML_TYPE_Q1_0},
    {"q2_0", GGML_TYPE_Q2_0},
    {"q4_0", GGML_TYPE_Q4_0},
    {"q4_1", GGML_TYPE_Q4_1},
    {"q5_0", GGML_TYPE_Q5_0},
    {"q5_1", GGML_TYPE_Q5_1},
    {"q8_0", GGML_TYPE_Q8_0},
    {"q2_k", GGML_TYPE_Q2_K},
    {"q3_k", GGML_TYPE_Q3_K},
    {"q4_k", GGML_TYPE_Q4_K},
    {"q5_k", GGML_TYPE_Q5_K},
    {"q6_k", GGML_TYPE_Q6_K},
    {"q8_k", GGML_TYPE_Q8_K},
    {"mxfp4", GGML_TYPE_MXFP4},
    {"nvfp4", GGML_TYPE_NVFP4},
};

ggml_type parse_format(std::string_view name) {
    const auto found = std::find_if(std::begin(formats), std::end(formats),
        [name](const format_entry& entry) { return entry.name == name; });
    if (found == std::end(formats)) {
        throw std::invalid_argument("unsupported ggml quantization format: " +
                                    std::string(name));
    }
    return found->type;
}

std::size_t element_count(std::span<const std::int64_t> shape) {
    if (shape.empty()) {
        throw std::invalid_argument("shape must have at least one dimension");
    }
    std::size_t count = 1;
    for (const auto dimension : shape) {
        if (dimension <= 0) {
            throw std::invalid_argument("ggml quantization requires positive dimensions");
        }
        const auto extent = static_cast<std::size_t>(dimension);
        if (extent > std::numeric_limits<std::size_t>::max() / count) {
            throw std::overflow_error("shape element count overflows size_t");
        }
        count *= extent;
    }
    return count;
}

std::int64_t row_count(std::size_t count, std::int64_t row_width) {
    if (count > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error("shape element count exceeds ggml's int64 range");
    }
    return static_cast<std::int64_t>(count) / row_width;
}

} // namespace

std::vector<std::string> supported_formats() {
    std::vector<std::string> result;
    result.reserve(std::size(formats));
    for (const auto& format : formats) {
        if (!ggml_quantize_requires_imatrix(format.type)) {
            result.emplace_back(format.name);
        }
    }
    return result;
}

quantized_values quantize(
    std::span<const float> values,
    std::span<const std::int64_t> shape,
    std::string_view format) {
    const auto count = element_count(shape);
    if (count != values.size()) {
        throw std::invalid_argument("shape extent does not match input value count");
    }
    if (!std::all_of(values.begin(), values.end(),
                     [](float value) { return std::isfinite(value); })) {
        throw std::invalid_argument("ggml quantization requires finite input values");
    }
    const auto type = parse_format(format);
    // NumPy's last dimension is contiguous in its standard row-major layout.
    const auto row_width = shape.back();
    const auto block = ggml_blck_size(type);
    if (row_width % block != 0) {
        throw std::invalid_argument("last-dimension row width must be a ggml block multiple");
    }
    if (ggml_quantize_requires_imatrix(type)) {
        throw std::invalid_argument("this format requires an explicit importance matrix");
    }

    const auto rows = row_count(count, row_width);
    const auto row_bytes = ggml_row_size(type, row_width);
    if (row_bytes > std::numeric_limits<std::size_t>::max() /
                    static_cast<std::size_t>(rows)) {
        throw std::overflow_error("quantized payload size overflows size_t");
    }

    quantized_values result;
    result.format.assign(format);
    result.shape.assign(shape.begin(), shape.end());
    result.payload.resize(row_bytes * static_cast<std::size_t>(rows));
    const auto written = ggml_quantize_chunk(type, values.data(), result.payload.data(),
                                             0, rows, row_width, nullptr);
    if (written != result.payload.size()) {
        throw std::runtime_error("ggml returned an unexpected quantized payload size");
    }
    return result;
}

std::vector<float> dequantize(const quantized_values& values) {
    const auto count = element_count(values.shape);
    const auto type = parse_format(values.format);
    const auto row_width = values.shape.back();
    if (row_width % ggml_blck_size(type) != 0) {
        throw std::invalid_argument("stored row width is not a ggml block multiple");
    }
    const auto rows = row_count(count, row_width);
    const auto row_bytes = ggml_row_size(type, row_width);
    if (row_bytes > std::numeric_limits<std::size_t>::max() /
                    static_cast<std::size_t>(rows)) {
        throw std::overflow_error("quantized payload size overflows size_t");
    }
    const auto expected = row_bytes * static_cast<std::size_t>(rows);
    if (values.payload.size() != expected) {
        throw std::invalid_argument("quantized payload length does not match shape and format");
    }
    const auto* traits = ggml_get_type_traits(type);
    if (traits == nullptr || traits->to_float == nullptr) {
        throw std::invalid_argument("ggml format has no float reconstruction function");
    }

    std::vector<float> result(count);
    for (std::int64_t row = 0; row < rows; ++row) {
        traits->to_float(values.payload.data() + static_cast<std::size_t>(row) * row_bytes,
                         result.data() + static_cast<std::size_t>(row) * row_width,
                         row_width);
    }
    return result;
}

} // namespace cellerator::interop::ggml
