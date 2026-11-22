#include <Cellerator/memory/view.hh>
#include <type_traits>

import ccell.preprocess;
import ccell.velocity;
import preprocess;
import velocitypp;
import velocitypp.precision;

static_assert(std::is_same_v<ccell::preprocess::QcStatus, preprocess::QcStatus>);
static_assert(std::is_same_v<ccell::velocity::PreparedInput, velocitypp::PreparedInput>);

#include "workflow.hh"

int main() { return workflow(); }
