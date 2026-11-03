#pragma once

#include "contract.hh"
#include <string>

namespace cgs {

// Private, complete-condition-to-update study entry point. All CUDA work must
// execute under the study's admitted Scheduler/Autobench job. A fresh process
// is used for each paired candidate; all arms receive the same case and seed.
struct CampaignOptions {
    std::string case_id;
    std::string replay_path;
    std::string backend = "scalar";
    std::string condition_update = "none"; // none, rebuild, incremental
    GateProfile gate = GateProfile::supplied;
    BodyKind body = BodyKind::ordered_product;
    DType dtype = DType::fp16;
    unsigned cell_tile = 32;
    unsigned stage_count = 1;
    unsigned warmups = 1;
    unsigned repeats = 5;
};

// The output is one cgs.campaign.v1 JSON record. The function returns nonzero
// on unsupported cases, failed reference checks, device failures, or I/O errors.
int run_campaign(const CampaignOptions&, const std::string& output_path);

// Host-only proof that rebuilt execution layouts are bit-identical to the
// independently generated fixture for each body and storage dtype. No CUDA
// context is created. An empty case selects the complete workload manifest.
bool run_campaign_layout_checks(const std::string& case_id, std::string* error = nullptr);

} // namespace cgs
