#pragma once
#include <Cellerator/ops/automata/dfa32.hh>
#include <Cellerator/ops/affine/monomial.hh>
#include <Cellerator/ops/sequence/lifting.hh>
#include <Cellerator/ops/reduction/residual_tree.hh>
#include <Cellerator/ops/relation/fixed16.hh>
#include <Cellerator/ops/numeric/weighted_relations.hh>
#include <Cellerator/ops/affine/block.hh>
#include <Cellerator/ops/response/quadratic_jet.hh>
#include <Cellerator/ops/sequence/monomial_checkpoints.hh>
#include <Cellerator/ops/numeric/precision_interval.hh>
#include <Cellerator/ops/matrix/host_reference.hh>
#include <Cellerator/ops/process/condensed_ports.hh>
#include <Cellerator/ops/iterative/coarse_correction.hh>
namespace cellerator::math::effects {
// Compatibility aliases retain the historical names while numerical ownership
// lives in operation-family headers under Cellerator/ops.
using ce_moon::Dfa32;
using ce_moon::CountedDfa32;
using ce_moon::MonomialAffine;
using ce_moon::LiftPair;
using ce_moon::ResidualTree;
using ce_moon::Relation16;
using ce_moon::compose;
using ce_moon::apply;
using ce_moon::compose_relation;
using ce_moon::lift;
using ce_moon::unlift;
using ce_moon::effects::Algebra;
using ce_moon::effects::Weighted;
using ce_moon::effects::BlockAffine;
using ce_moon::effects::Jet;
using ce_moon::effects::JetQuery;
using ce_moon::effects::Checkpoints;
using ce_moon::effects::compose;
using ce_moon::effects::apply;
using ce_moon::effects::query;
using ce_moon::mechanisms::Matrix;
using ce_moon::mechanisms::PortResponse;
using ce_moon::mechanisms::condense_ports;
using ce_moon::mechanisms::compose_port_system;
using ce_moon::mechanisms::solve_ports;
using ce_moon::mechanisms::reconstruct_interior;
} // namespace cellerator::math::effects
