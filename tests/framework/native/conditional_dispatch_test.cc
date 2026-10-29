#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/execution/conditional/dispatch.hh>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace cellerator::execution::conditional;

namespace {
bool near(double a, double b, double eps = 1e-10) {
    return std::abs(a - b) <= eps * (1.0 + std::abs(a) + std::abs(b));
}

// Independent oracle: these loops use the caller's selection sequence and do
// not use the plan's grouping or inverse maps.
std::vector<double> oracle_combine(const std::vector<Selection> &routes,
                                   const std::vector<double> &dispatch,
                                   std::size_t features) {
    std::vector<Origin> origins;
    for (const auto &route : routes) {
        bool seen = false;
        for (const auto &origin : origins) seen = seen || origin == route.origin;
        if (!seen) origins.push_back(route.origin);
    }
    std::vector<double> out(origins.size() * features, 0.0);
    for (std::size_t i = 0; i < routes.size(); ++i) {
        std::size_t origin_index = 0;
        while (!(origins[origin_index] == routes[i].origin)) ++origin_index;
        for (std::size_t f = 0; f < features; ++f)
            out[origin_index * features + f] += routes[i].weight * dispatch[i * features + f];
    }
    return out;
}

std::vector<double> oracle_transpose(const std::vector<Selection> &routes,
                                     const std::vector<double> &origin_grad,
                                     std::size_t features) {
    std::vector<Origin> origins;
    for (const auto &route : routes) {
        bool seen = false;
        for (const auto &origin : origins) seen = seen || origin == route.origin;
        if (!seen) origins.push_back(route.origin);
    }
    std::vector<double> out(routes.size() * features);
    for (std::size_t i = 0; i < routes.size(); ++i) {
        std::size_t origin_index = 0;
        while (!(origins[origin_index] == routes[i].origin)) ++origin_index;
        for (std::size_t f = 0; f < features; ++f)
            out[i * features + f] = routes[i].weight * origin_grad[origin_index * features + f];
    }
    return out;
}

void expect_invalid(std::vector<Selection> routes) {
    bool thrown = false;
    try { (void)DispatchPlan(std::move(routes)); }
    catch (const std::invalid_argument &) { thrown = true; }
    assert(thrown);
}
} // namespace

int main() {
    const ExpertKey e1{10u, 1u};
    const ExpertKey e1_new{10u, 2u}; // same ID, distinct revision
    const ExpertKey e2{20u, 7u};
    const ExpertKey e3{30u, 1u};
    const Origin a{0u, 100u, 5u};
    const Origin b{1u, 100u, 5u}; // same instance/row, distinct rank
    const Origin c{1u, 200u, 5u}; // same rank/row, distinct instance
    const std::vector<Selection> logical{
        {e1, a, 2.0}, {e2, b, -0.5}, {e1, a, 0.0},
        {e1_new, b, 1.0}, {e2, c, 0.0}, {e3, b, 3.0},
    };
    const DispatchPlan plan(logical);
    assert(plan.size() == logical.size());
    assert(plan.origin_count() == 3u);
    assert((plan.origins() == std::vector<Origin>{a, b, c}));
    assert(plan.experts().size() == 4u);
    assert(plan.expert_offsets().size() == plan.experts().size() + 1u);
    assert(plan.origin_offsets().size() == plan.origins().size() + 1u);
    assert(plan.dispatch_to_selection().size() == logical.size());
    assert(plan.selection_to_dispatch().size() == logical.size());
    assert(plan.selection_to_origin().size() == logical.size());

    for (std::size_t dispatch = 0; dispatch < plan.size(); ++dispatch) {
        const auto selection = plan.dispatch_to_selection()[dispatch];
        assert(selection < logical.size());
        assert(plan.selection_to_dispatch()[selection] == dispatch);
        assert(plan.selections()[selection].expert == logical[selection].expert);
        assert(plan.selections()[selection].origin == logical[selection].origin);
        assert(plan.selections()[selection].weight == logical[selection].weight);
    }
    for (std::size_t expert = 0; expert < plan.experts().size(); ++expert) {
        assert(plan.expert_offsets()[expert] <= plan.expert_offsets()[expert + 1u]);
        for (std::size_t i = plan.expert_offsets()[expert];
             i < plan.expert_offsets()[expert + 1u]; ++i)
            assert(plan.selections()[plan.dispatch_to_selection()[i]].expert == plan.experts()[expert]);
    }
    std::vector<std::size_t> origin_cursor(plan.origin_count());
    for (std::size_t origin = 0; origin < plan.origin_count(); ++origin)
        origin_cursor[origin] = plan.origin_offsets()[origin];
    for (std::size_t selection = 0; selection < logical.size(); ++selection) {
        const auto origin = plan.selection_to_origin()[selection];
        assert(plan.origins()[origin] == logical[selection].origin);
        assert(plan.origin_dispatch_indices()[origin_cursor[origin]++]
            == plan.selection_to_dispatch()[selection]);
    }

    constexpr std::size_t in_features = 2u;
    constexpr std::size_t out_features = 2u;
    std::size_t provider_calls = 0u;
    const auto input = gather_inputs<double>(plan, in_features,
        [&](const Origin &origin, double *row, std::size_t count) {
            ++provider_calls;
            assert(count == 2u);
            row[0] = 100.0 * origin.rank + 10.0 * origin.instance + origin.row;
            row[1] = row[0] + 1.0;
        });
    assert(provider_calls == plan.origin_count());
    std::vector<double> logical_input(logical.size() * in_features);
    for (std::size_t i = 0; i < logical.size(); ++i) {
        const auto &origin = logical[i].origin;
        logical_input[2 * i] = 100.0 * origin.rank + 10.0 * origin.instance + origin.row;
        logical_input[2 * i + 1] = logical_input[2 * i] + 1.0;
    }
    for (std::size_t dispatch = 0; dispatch < plan.size(); ++dispatch) {
        const auto selection = plan.dispatch_to_selection()[dispatch];
        assert(input[2 * dispatch] == logical_input[2 * selection]);
        assert(input[2 * dispatch + 1] == logical_input[2 * selection + 1]);
    }

    std::vector<double> values(plan.size() * out_features);
    std::vector<double> logical_values(logical.size() * out_features);
    for (std::size_t selection = 0; selection < logical.size(); ++selection)
        for (std::size_t f = 0; f < out_features; ++f)
            logical_values[selection * out_features + f] =
                0.25 + static_cast<double>(selection * out_features + f);
    for (std::size_t dispatch = 0; dispatch < plan.size(); ++dispatch)
        for (std::size_t f = 0; f < out_features; ++f)
            values[dispatch * out_features + f] = logical_values[
                plan.dispatch_to_selection()[dispatch] * out_features + f];
    const auto combined = combine<double, double, double>(plan, values, out_features);
    const auto expected = oracle_combine(logical, logical_values, out_features);
    assert(combined.size() == expected.size());
    for (std::size_t i = 0; i < combined.size(); ++i) assert(near(combined[i], expected[i]));

    const std::vector<double> grad{0.5, -2.0, 4.0, 3.0, -1.0, 7.0};
    const auto transposed = transpose_combine<double, double, double>(plan, grad, out_features);
    const auto expected_transpose = oracle_transpose(logical, grad, out_features);
    for (std::size_t dispatch = 0; dispatch < plan.size(); ++dispatch)
        for (std::size_t f = 0; f < out_features; ++f)
            assert(near(transposed[dispatch * out_features + f],
                expected_transpose[plan.dispatch_to_selection()[dispatch] * out_features + f]));

    // Weighted forward combination and transpose must satisfy the dot-product
    // identity, including negative, zero, and repeated route weights.
    double lhs = 0.0, rhs = 0.0;
    const auto forward = combine<double, double, double>(plan, values, out_features);
    for (std::size_t i = 0; i < forward.size(); ++i) lhs += forward[i] * grad[i];
    for (std::size_t i = 0; i < transposed.size(); ++i) rhs += values[i] * transposed[i];
    assert(near(lhs, rhs));

    std::vector<std::size_t> callback_visits(logical.size(), 0u);
    const auto executed = execute<double, double, double, double>(
        plan, in_features, out_features,
        [](const Origin &origin, double *row, std::size_t) {
            row[0] = 1.0 + origin.instance + origin.row;
            row[1] = row[0] * 2.0;
        },
        [&](const ExpertKey &expert, const InputBatch<double> &in, OutputBatch<double> &out) {
            assert(in.rows == out.rows);
            assert(out.features == 2u);
            const double scale = static_cast<double>(expert.expert_id) + expert.revision * 0.1;
            for (std::size_t row = 0; row < in.rows; ++row) {
                const auto selection = in.selection_indices[row];
                assert(selection < logical.size());
                assert(logical[selection].expert == expert);
                assert(logical[selection].origin == in.origins[row]);
                assert(in.at(row, 0u) == 1.0 + in.origins[row].instance + in.origins[row].row);
                assert(in.at(row, 1u) == 2.0 * in.at(row, 0u));
                ++callback_visits[selection];
                out.at(row, 0u) = scale * in.at(row, 0u);
                out.at(row, 1u) = scale * in.at(row, 1u) - 2.0;
            }
        });
    for (const auto visits : callback_visits) assert(visits == 1u);
    assert(executed.size() == plan.origin_count() * out_features);
    std::vector<double> expected_execution(plan.origin_count() * out_features, 0.0);
    for (const auto &route : logical) {
        std::size_t origin_index = 0u;
        while (!(plan.origins()[origin_index] == route.origin)) ++origin_index;
        const double x0 = 1.0 + route.origin.instance + route.origin.row;
        const double x1 = 2.0 * x0;
        const double scale = static_cast<double>(route.expert.expert_id)
            + route.expert.revision * 0.1;
        expected_execution[origin_index * out_features] += route.weight * scale * x0;
        expected_execution[origin_index * out_features + 1u] +=
            route.weight * (scale * x1 - 2.0);
    }
    for (std::size_t i = 0; i < executed.size(); ++i)
        assert(near(executed[i], expected_execution[i]));

    // Sorting these expert groups would sum -1e16, +1, +1e16 and round to
    // zero. The contract retains caller occurrence order, which sums to one.
    const Origin cancellation_origin{9u, 9u, 9u};
    const DispatchPlan cancellation({
        {{30u, 1u}, cancellation_origin, 1.0},
        {{10u, 1u}, cancellation_origin, 1.0},
        {{20u, 1u}, cancellation_origin, 1.0},
    });
    const std::vector<double> cancellation_values_by_selection{1.0e16, -1.0e16, 1.0};
    std::vector<double> cancellation_dispatch(cancellation.size());
    for (std::size_t dispatch = 0; dispatch < cancellation.size(); ++dispatch)
        cancellation_dispatch[dispatch] = cancellation_values_by_selection[
            cancellation.dispatch_to_selection()[dispatch]];
    const auto cancellation_sum = combine<double>(cancellation, cancellation_dispatch, 1u);
    assert(cancellation_sum.size() == 1u && cancellation_sum[0] == 1.0);
    // The empty route remains a valid, zero-work plan.
    const DispatchPlan empty({});
    assert(empty.size() == 0u && empty.origin_count() == 0u);
    assert(combine<double>(empty, {}, out_features).empty());
    assert(transpose_combine<double>(empty, {}, out_features).empty());
    assert(gather_inputs<double>(empty, std::numeric_limits<std::size_t>::max(),
        [](const Origin&, double*, std::size_t) { assert(false); }).empty());
    bool output_shape_rejected = false;
    try {
        (void)execute<double>(plan, in_features, 0,
            [](const Origin&, double*, std::size_t) { assert(false); },
            [](const ExpertKey&, const InputBatch<double>&, OutputBatch<double>&) { assert(false); });
    } catch (const std::invalid_argument&) { output_shape_rejected = true; }
    assert(output_shape_rejected);

    expect_invalid({{e1, a, std::numeric_limits<double>::quiet_NaN()}});
    expect_invalid({{e1, a, std::numeric_limits<double>::infinity()}});
    bool bad_shape = false;
    try { (void)combine<double>(plan, {1.0}, out_features); }
    catch (const std::invalid_argument &) { bad_shape = true; }
    assert(bad_shape);
    bad_shape = false;
    try { (void)transpose_combine<double>(plan, {1.0}, out_features); }
    catch (const std::invalid_argument &) { bad_shape = true; }
    assert(bad_shape);
    bool propagated = false;
    try {
        (void)execute<double>(plan, in_features, out_features,
            [](const Origin &, double *row, std::size_t) { row[0] = row[1] = 1.0; },
            [](const ExpertKey &, const InputBatch<double> &, OutputBatch<double> &) {
                throw std::runtime_error("expert failed");
            });
    } catch (const std::runtime_error &) { propagated = true; }
    assert(propagated);
    return 0;
}
