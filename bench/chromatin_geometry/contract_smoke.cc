#include "contract.hh"

#include <cstdlib>
#include <iostream>

static_assert(static_cast<unsigned>(cgs::Backend::scalar) == 0);
static_assert(static_cast<unsigned>(cgs::Backend::bitmap) == 1);
static_assert(static_cast<unsigned>(cgs::Backend::direct) == 2);
static_assert(static_cast<unsigned>(cgs::Backend::cohort_direct) == 3);
static_assert(static_cast<unsigned>(cgs::Backend::cohort_shared) == 4);
static_assert(static_cast<unsigned>(cgs::Backend::cohort_direct_two_tile) == 5);
static_assert(static_cast<unsigned>(cgs::Backend::masked_direct_two_tile) == 6);

int main() {
    cgs::GateRevisions a{}, b{};
    if (!cgs::same_revisions(a, b)) return EXIT_FAILURE;
    b.primal_generation = 1;
    if (cgs::same_revisions(a, b)) return EXIT_FAILURE;
    b = a;
    b.derivative_generation = 1;
    if (cgs::same_revisions(a, b)) return EXIT_FAILURE;
    b = a;
    b.temperature_bits = 0x3f800000u;
    if (cgs::same_revisions(a, b)) return EXIT_FAILURE;
    b = a;
    b.threshold_bits = 0x3f000000u;
    if (cgs::same_revisions(a, b)) return EXIT_FAILURE;
    b = a;
    b.mapping_revision = 1;
    if (cgs::same_revisions(a, b)) return EXIT_FAILURE;
    const cgs::ProblemView p{};
    if (p.batch != 0 || p.include_baseline || p.dtype != cgs::DType::fp16)
        return EXIT_FAILURE;
    std::cout << "cgs contract smoke: passed\n";
    return EXIT_SUCCESS;
}
