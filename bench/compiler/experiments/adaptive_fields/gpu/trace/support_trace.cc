#include <Cellerator/compute/support/adaptive_support.hh>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace support = cellerator::compute::support;

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

const char* status_name(support::adaptive_support_status_v1 status) {
  using S = support::adaptive_support_status_v1;
  switch (status) {
    case S::success: return "success";
    case S::live_forward_tapes: return "live_forward_tapes";
    case S::stale_epoch: return "stale_epoch";
    default: return "other_error";
  }
}

struct observed_workset {
  std::uint64_t structure_epoch = 0;
  std::uint64_t support_count = 0;
  std::vector<std::uint64_t> ids;
};

observed_workset query_source(support::adaptive_support_manager_v1& manager,
                              std::uint64_t source_entity_id) {
  support::support_snapshot_pair_v1 pair{};
  require(manager.snapshot(&pair) == support::adaptive_support_status_v1::success,
          "snapshot failed");
  observed_workset result;
  result.structure_epoch = pair.snapshot_structure_epoch;
  auto visit = [&](const support::logical_support_view_v1* relation) {
    if (!relation) return;
    result.support_count += relation->edge_count;
    for (std::uint64_t ordinal = 0; ordinal < relation->edge_count; ++ordinal) {
      const auto& edge = relation->edges[ordinal];
      const auto& source = relation->participants[edge.participant_begin];
      if (source.role == 0u && source.entity_id == source_entity_id)
        result.ids.push_back(edge.logical_edge_id);
    }
  };
  visit(pair.base);
  visit(pair.delta);
  std::sort(result.ids.begin(), result.ids.end());
  return result;
}

void emit_ids(const std::vector<std::uint64_t>& ids) {
  std::cout << '[';
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << ids[i];
  }
  std::cout << ']';
}
void emit_entities(bool all_sources) {
  std::cout << '[';
  if (all_sources) {
    for (std::uint64_t source = 1; source <= 16; ++source) {
      if (source != 1) std::cout << ',';
      std::cout << source;
    }
  } else {
    std::cout << '1';
  }
  std::cout << ']';
}
}  // namespace

int main() {
  try {
    constexpr std::uint64_t kInitialEpoch = 1;
    constexpr std::uint64_t kNextEpoch = 2;
    constexpr std::uint64_t kWindowSteps = 16;
    constexpr std::uint64_t kSourceEntities = 16;

    std::vector<support::logical_support_edge_v1> edges;
    std::vector<support::logical_support_participant_v1> participants;
    edges.reserve(64);
    participants.reserve(128);
    for (std::uint64_t i = 0; i < 64; ++i) {
      const std::uint64_t source = i % kSourceEntities;
      const std::uint64_t lane = i / kSourceEntities;
      const std::uint64_t destination = (source + 1 + lane) % kSourceEntities;
      edges.push_back({i + 1, i * 2, 2, 0});
      participants.push_back({0, 0, source + 1});
      participants.push_back({1, 0, destination + 1});
    }
    support::logical_support_view_v1 initial{
        support::logical_support_schema_version_v1, 0, 8101, 8102,
        kInitialEpoch, edges.data(), edges.size(), participants.data(),
        participants.size()};
    support::adaptive_support_manager_v1 manager;
    require(manager.initialize(initial, 8103)
                == support::adaptive_support_status_v1::success,
            "adaptive support initialization failed");

    std::uint64_t state_generation = 1;
    std::cout << "{\"schema\":\"cellerator.adaptive_fields.e7_support_trace.v1\",\"steps\":[\n";
    bool first_step = true;
    // Start a low-density window. Its pinned owner state is retired before
    // publication; publication is also deliberately attempted while pinned.
    require(manager.pin_forward_tape(kInitialEpoch)
                == support::adaptive_support_status_v1::success,
            "failed to pin first training window");
    for (std::uint64_t step = 0; step < kWindowSteps; ++step) {
      const auto row = query_source(manager, 1);
      if (!first_step) std::cout << ",\n";
      first_step = false;
      std::cout << "{\"phase\":\"low_before_publication\",\"step\":"
                << state_generation - 1 << ",\"structure_epoch\":"
                << row.structure_epoch << ",\"state_generation\":"
                << state_generation++ << ",\"support_count\":"
                << row.support_count << ",\"logical_edge_ids\":";
      emit_ids(row.ids);
      std::cout << ",\"query_source_entities\":[1]}";
    }

    const support::logical_support_participant_v1 promoted_participants[] = {
        {0, 0, 1}, {1, 0, 5}};
    const support::staged_support_edge_v1 proposal{
        100, promoted_participants, 2, 0, 17, 3, 0.875f};
    require(manager.stage_promotion(proposal)
                == support::adaptive_support_status_v1::success,
            "support promotion staging failed");
    const auto pinned_publish = manager.publish_pending(kNextEpoch);
    require(pinned_publish == support::adaptive_support_status_v1::live_forward_tapes,
            "publication was not blocked by the live training window");
    require(manager.retire_forward_tape(kInitialEpoch)
                == support::adaptive_support_status_v1::success,
            "failed to retire first training window");
    const auto publish_status = manager.publish_pending(kNextEpoch);
    require(publish_status == support::adaptive_support_status_v1::success,
            "support publication failed after tape retirement");

    // Emit the two post-publication windows and preserve the exact identities
    // observed by the production logical support manager.
    auto emit_phase = [&](const char* phase, bool all_sources) {
      const auto epoch = manager.structure_epoch();
      require(manager.pin_forward_tape(epoch)
                  == support::adaptive_support_status_v1::success,
              "failed to pin post-publication training window");
      for (std::uint64_t step = 0; step < kWindowSteps; ++step) {
        observed_workset row;
        if (all_sources) {
          row.structure_epoch = epoch;
          for (std::uint64_t source = 0; source < kSourceEntities; ++source) {
            auto one = query_source(manager, source + 1);
            row.support_count = one.support_count;
            row.ids.insert(row.ids.end(), one.ids.begin(), one.ids.end());
          }
          std::sort(row.ids.begin(), row.ids.end());
        } else {
          row = query_source(manager, 1);
        }
        if (!first_step) std::cout << ",\n";
        first_step = false;
        std::cout << "{\"phase\":\"" << phase << "\",\"step\":"
                  << state_generation - 1 << ",\"structure_epoch\":"
                  << row.structure_epoch << ",\"state_generation\":"
                  << state_generation++ << ",\"support_count\":"
                  << row.support_count << ",\"logical_edge_ids\":";
        emit_ids(row.ids);
        std::cout << ",\"query_source_entities\":";
        emit_entities(all_sources);
        std::cout << "}";
      }
      require(manager.retire_forward_tape(epoch)
                  == support::adaptive_support_status_v1::success,
              "failed to retire post-publication training window");
    };
    emit_phase("high_after_publication", true);
    emit_phase("low_after_publication", false);
    std::cout << "],\"publication\":{\"pinned_attempt\":\""
              << status_name(pinned_publish) << "\",\"status\":\""
              << status_name(publish_status) << "\",\"structure_epoch\":"
              << manager.structure_epoch() << ",\"promoted_logical_edge_id\":100,\"state_generation\":"
              << state_generation - 1 << "}}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "support_trace_error: " << error.what() << "\n";
    return 1;
  }
}
