#include <Cellerator/compute/operation/indexed_mechanism/evaluators.hh>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace ix = cellerator::compute::operation::indexed;
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
int main() try {
    ix::registered_block composed{{300, 1}, ix::evaluator_opcode::first_minus_product_tail, 3};
    ix::registered_block custom{{301, 1}, ix::evaluator_opcode::first_minus_product_tail, 3};
    ix::block_registry registry;
    check(registry.register_block(composed) == ix::evaluation_status::success, "composed block registered");
    check(registry.register_block(custom) == ix::evaluation_status::success, "custom block registered");
    check(registry.register_block(custom) == ix::evaluation_status::duplicate_block, "duplicate custom block rejected");
    check(registry.find(custom.evaluator) != nullptr, "custom lookup returns executable block");

    std::array<float, 5> arguments{10.0f, 2.0f, 3.0f, 2.0f, 1.0f};
    std::array<float, 1> composed_output{-7.0f}, custom_output{-9.0f};
    check(ix::evaluate_f32(composed, true, arguments, composed_output) == ix::evaluation_status::success, "composed evaluation");
    check(ix::evaluate_f32(*registry.find(custom.evaluator), true, arguments, custom_output) == ix::evaluation_status::success, "custom evaluation");
    check(composed_output[0] == -2.0f && custom_output == composed_output, "nonadditive runtime-width custom/composed parity");

    const std::array<float, 3> invalid{std::numeric_limits<float>::quiet_NaN(), 4.0f, 5.0f};
    std::array<float, 1> masked{17.0f};
    check(ix::evaluate_f32(composed, false, invalid, masked) == ix::evaluation_status::success && masked[0] == 17.0f,
          "false predicate excludes nonfinite operand and preserves destination");
    check(ix::evaluate_f32(composed, true, invalid, masked) == ix::evaluation_status::success && std::isnan(masked[0]),
          "active nonfinite is never silently sanitized");

    std::array<std::uint16_t, 3> half_arguments{0x4900u, 0x4000u, 0x4600u}; // 10, 2, 6
    std::array<std::uint16_t, 1> half_output{};
    check(ix::evaluate_f16(composed, true, half_arguments, half_output) == ix::evaluation_status::success && half_output[0] == 0xc000u,
          "FP16 storage uses FP32 baseline arithmetic and RNE result");
    std::cout << "H02 registered composition/custom parity, predicate exclusion, FP32 and FP16 host routes passed\n";
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
