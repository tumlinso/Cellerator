#include <ce_moon/reference.hpp>
#include <ce_moon/effects.hpp>
int main() {
    const auto a=ce_moon::Dfa32::identity();
    const auto combined=ce_moon::compose(a,a);
    return combined.to[17]==17 ? 0 : 1;
}
