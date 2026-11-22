#include <Cellerator/math/effects/affine.hh>

namespace fx = cellerator::math::effects;
namespace ex = cellerator::execution;

int main() {
    const fx::rel::axis_descriptor axis{{{ex::biological_abi_version,
        ex::serialized_record_kind::persistent_axis_identity, sizeof(ex::persistent_axis_identity)},
        {1, 1}, {2, 1}, {3, 1}, {4, 1}}, 1};
    auto a = fx::accumulated_affine<1, 1>::identity({axis, axis, {1}});
    a.A[0][0] = 2; a.b[0] = 1; a.C[0][0] = 3;
    const auto result = fx::compose(a, a).apply({4}, {0});
    return result.first[0] == 19 && result.second[0] == 39 ? 0 : 1;
}
