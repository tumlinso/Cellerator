#pragma once
#include <Cellerator/compute/operation/product2/product2.hh>
namespace cellerator::math::process {
// M06's integrated native owner: no duplicate topology or coefficients.
using prepared_product=compute::product2::prepared_owner;
using product_descriptor=compute::product2::descriptor;
using product_binding=compute::product2::bound_view;
using product_action=compute::product2::operation;
} // namespace cellerator::math::process
