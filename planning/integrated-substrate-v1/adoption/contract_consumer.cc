#include <Cellerator/compute/operation/host_contracts.hh>
#include <type_traits>
namespace nf = cellerator::compute::operation::nf1;
namespace ix = cellerator::compute::operation::indexed;
namespace ex = cellerator::execution;
static_assert(std::is_same_v<ix::identity, nf::identity>);
static_assert(!std::is_same_v<ex::structure_epoch, ex::value_generation>);
static_assert(nf::contract_version == 1);
static_assert(cellerator::compute::native_numeric::local_arity(
    cellerator::compute::native_numeric::local_operation::multiply) == 2);
int main() {
    // Empty semantics is rejected by the actual owner, with its native status.
    const auto result = cellerator::compute::relation::validate({});
    return result ? 1 : 0;
}
