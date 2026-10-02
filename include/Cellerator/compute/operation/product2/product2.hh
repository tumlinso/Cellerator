#pragma once
#include "Cellerator/compute/operation/product2/c_api.h"
#include "Cellerator/execution/program/program_v2.h"
#include <vector>
#include "Cellerator/compute/operation/indexed_mechanism/incidence.hh"
namespace cellerator::compute::product2 {
enum class operation { forward, vjp, jvp };
struct descriptor { uint64_t input_count=0, product_count=0, structure_generation=0; const int64_t *a=nullptr,*b=nullptr; uint64_t a_count=0,b_count=0; bool binds_axes=false;
 ::cellerator::compute::operation::indexed::indexed_axis input_axis{},product_axis{},coefficient_axis{}; };
struct bound_view { ce_product2_binding numeric{}; execution::persistent_axis_identity input_axis{},product_axis{},coefficient_axis{}; };
class prepared_owner {
 public:
  explicit prepared_owner(const descriptor&);
  prepared_owner(const prepared_owner&)=delete;
  prepared_owner& operator=(const prepared_owner&)=delete;
  prepared_owner(prepared_owner&&)=delete;
  prepared_owner& operator=(prepared_owner&&)=delete;
  const descriptor& topology() const noexcept {return topology_;}
  ce_product2_status admit(const ce_product2_binding&,operation) const noexcept;
  ce_product2_status admit(const bound_view&,operation) const noexcept;
  ce_product2_status execute(const ce_product2_binding&,operation) const noexcept;
  ce_product2_status execute(const bound_view&,operation) const noexcept;
 private:
  ce_product2_status admit_numeric(const ce_product2_binding&,operation) const noexcept;
  ce_product2_status execute_numeric(const ce_product2_binding&,operation) const noexcept;
  std::vector<int64_t> a_,b_;
  descriptor topology_;
};
// The stage state and binding object must outlive execute_prepared_program_v2.
// A bound_view is supplied through launch_binding_v2::input; the existing ABI
// performs admission of every stage before any callback launches.
struct stage_state { const prepared_owner *owner=nullptr; operation op=operation::forward; };
execution::program::prepared_stage_v2 make_stage(const stage_state&,uint64_t stable_id,uint32_t binding_index) noexcept;
#ifdef CELLERATOR_PRODUCT2_HAS_CUDA
struct cuda_stage_state { const prepared_owner *owner=nullptr; const ce_product2_cuda_context *context=nullptr; operation op=operation::forward; };
execution::program::prepared_stage_v2 make_cuda_stage(const cuda_stage_state&,uint64_t stable_id,uint32_t binding_index) noexcept;
#endif
} // namespace
