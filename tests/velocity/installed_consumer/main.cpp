#include <cstdint>

import anndatapp.h5ad;
import cellerator.core;
import cellerator.velocity;
import velocitypp;

struct Model {
    int operator()(int& state, int increment) const { return state + increment; }
};

int main() {
    using namespace cellerator::velocity;

    const uint16_t s_values[]{0x3c00, 0x4000, 0x4200};
    const uint16_t u_values[]{0x4000, 0x4400, 0x4600};
    const uint16_t w_values[]{0x3c00, 0, 0, 0, 0x3c00, 0, 0, 0, 0x3c00};
    HalfMatrix s{anndatapp::Layout::dense, s_values, nullptr, nullptr,
        anndatapp::DType::int32, anndatapp::DType::int64, 3, 1, 0, 2, 2};
    HalfMatrix u = s;
    u.values = u_values;
    HalfMatrix w{anndatapp::Layout::dense, w_values, nullptr, nullptr,
        anndatapp::DType::int32, anndatapp::DType::int64, 3, 3, 0, 6, 2};
    Input input{{&s, nullptr}, {&u, nullptr}, {&w, nullptr}, {3, 1, 1}};
    if (!validate(input)) return 1;

    float ms[3], mu[3], source_s[3], source_u[3];
    moments(input, 0, 1, ms, mu, source_s, source_u);
    for (int i = 0; i < 3; ++i)
        if (ms[i] != float(i + 1) || mu[i] != 2 * ms[i]) return 2;

    Model model;
    int state = 37;
    auto cell = cellerator::make_cell(model, state);
    return cell.forward(5) == 42 ? 0 : 3;
}
