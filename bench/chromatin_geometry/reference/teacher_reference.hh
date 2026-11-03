#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace cgs::reference {

struct TeacherExampleReference final {
    float probability = 0.0f;
    float bce = 0.0f;
    float dlogit = 0.0f;
};

struct TeacherReference final {
    std::array<float, 8> probability{};
    std::array<std::uint8_t, 8> decision{};
    std::uint8_t table = 0;
    float mean_bce = 0.0f;
    std::array<float, 8> mean_logit_gradient{};
    std::array<float, 8> per_row_bce{};
    std::array<std::uint8_t, 8> row_has_examples{};
};

// Input bits use row=4*a+2*b+c. The labels are Bernoulli targets for the
// independent eight-row teacher. The returned hard table is the exact
// probability>=0.5 decision table; no task gradient is defined through it.
TeacherReference teacher_reference(const std::array<float, 8>& logits,
                                   const std::array<float, 8>& labels);

TeacherExampleReference teacher_example_reference(float logit,
                                                   float bernoulli_label);

// Average loss and per-row logit gradient over arbitrary sampled rows. This
// is the independent oracle for native minibatch teacher updates.
TeacherReference teacher_batch_reference(
    const std::array<float, 8>& logits,
    const std::uint8_t* rows, const float* labels, std::size_t count);

// Existing E29 fits one target per truth-table row with independent
// sigmoid(logit)-target updates. This minibatch extension first averages the
// BCE target within each observed row; empty rows have zero gradient and are
// left unchanged. Each nonempty row retains the original per-logit learning
// rate independently of how many samples landed in other rows.
TeacherReference teacher_row_mean_reference(
    const std::array<float, 8>& logits,
    const std::array<std::uint64_t, 8>& row_counts,
    const std::array<double, 8>& positive_counts);

std::uint8_t teacher_lookup(std::uint8_t table, bool a, bool b, bool c) noexcept;

} // namespace cgs::reference
