#include <Cellerator/execution/conditional/dispatch.hh>

#include <cstddef>
#include <cstdint>
#include <iostream>

using namespace cellerator::execution::conditional;

int main() {
    const ExpertKey growth_model{101u, 3u};
    const ExpertKey stress_model{202u, 1u};
    const Origin first_cell{0u, 17u, 4u};
    const Origin second_cell{0u, 17u, 9u};

    // An opaque model revision serves several cells, and two models contribute
    // to the first cell. The caller supplies route weights and semantics.
    const DispatchPlan plan({
        {growth_model, first_cell, 1.0},
        {stress_model, first_cell, 0.25},
        {growth_model, second_cell, 1.0},
    });
    const auto result = execute<double, double, double, double>(
        plan, 2u, 1u,
        [](const Origin &origin, double *row, std::size_t) {
            row[0] = static_cast<double>(origin.row);
            row[1] = static_cast<double>(origin.rank + origin.instance);
        },
        [](const ExpertKey &expert, const InputBatch<double> &inputs,
           OutputBatch<double> &outputs) {
            const double multiplier = expert.expert_id == 101u ? 0.5 : 2.0;
            for (std::size_t row = 0; row < inputs.rows; ++row)
                outputs.at(row, 0u) = multiplier *
                    (inputs.at(row, 0u) + inputs.at(row, 1u));
        });

    for (std::size_t i = 0; i < plan.origin_count(); ++i) {
        const auto &origin = plan.origins()[i];
        std::cout << "origin(rank=" << origin.rank
                  << ", instance=" << origin.instance
                  << ", row=" << origin.row << ") -> "
                  << result[i] << '\n';
    }
}
