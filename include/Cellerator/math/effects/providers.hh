#pragma once
#include <ce_moon/reference.hpp>
#include <ce_moon/effects.hpp>
#include <ce_moon/mechanisms.hpp>
namespace cellerator::math::effects {
// These are the original CE types/functions, not copies of their implementations.
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
