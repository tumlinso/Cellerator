#pragma once
#include "contract.hh"
#include <string>
namespace cgs {
struct BodyRunOptions { std::string case_id, replay_path; Backend backend=Backend::scalar; BodyKind body=BodyKind::ordered_product; DType dtype=DType::fp16; GateProfile gate=GateProfile::supplied; unsigned warmups=2, repeats=11; std::string condition_update="none"; };
// Isolated precomputed-gate numerical validation only, not a full campaign.
int run_body_check(const BodyRunOptions&, const std::string& output);
}
