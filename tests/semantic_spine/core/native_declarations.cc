#include <Cellerator/compute/operation/prepared_relation.hh>
#include <type_traits>
namespace r = cellerator::compute::relation;
namespace e = cellerator::execution;
using prepare_signature = r::status (*)(const r::operation_descriptor&,
    const r::operation_descriptor&, const r::csr_host_view&,
    const r::preparation_options&, cudaStream_t, r::prepared_relation_pair**) noexcept;
using enqueue_signature = r::status (*)(r::prepared_relation_pair&,
    const r::operation_descriptor&, const r::device_state_view&,
    const r::device_result_view&, e::value_generation, cudaStream_t) noexcept;
static_assert(std::is_same<decltype(&r::prepare_relation_pair), prepare_signature>::value);
static_assert(std::is_same<decltype(&r::enqueue), enqueue_signature>::value);
static_assert(std::is_trivially_copyable<r::device_values_binding>::value);
