#include <Cellerator/geometry/geometry.hh>

int main() {
    const auto readiness =
            cellerator::geometry::compiler::validate_integrated_optimizer_portfolio_v1();
    return readiness.contract_fingerprint == 0xe6b63b45ee0d35f9ULL
            && readiness.validated_strategies == 4
            && readiness.device_assisted_available
            && readiness.device_assisted_experimental ? 0 : 1;
}
