#pragma once
#include <Cellerator/compiler/ir/realization/implement_prepared_stage_graphs_v1.hh>
#include <cstdint>
#include <string>
#include <vector>
namespace cellerator::compiler::ir::realization::v1 {
enum class stream_class_v1:std::uint8_t{caller=1,compute,transfer,collective};
enum class synchronization_kind_v1:std::uint8_t{readiness_token=1,event_wait,transfer,device_link,host_synchronize};
struct launch_dependency_v1{std::uint32_t producer=0,consumer=0;stream_class_v1 producer_stream=stream_class_v1::compute,consumer_stream=stream_class_v1::compute;synchronization_kind_v1 kind=synchronization_kind_v1::readiness_token;stable_identity_v1 token{};std::uint32_t source_device=0,destination_device=0;bool explicit_host_sync=false;};
struct launch_dependency_graph_v1{std::uint32_t stage_count=0;std::vector<launch_dependency_v1>dependencies;std::uint32_t same_stream_elisions=0;};
enum class launch_dependency_status_v1:std::uint8_t{valid=0,invalid_stage,cycle,invalid_token,redundant_wait,implicit_host_sync,invalid_link};
[[nodiscard]] launch_dependency_status_v1 validate_launch_dependency_graph_v1(const launch_dependency_graph_v1&,std::string*error=nullptr) noexcept;
[[nodiscard]] launch_dependency_graph_v1 elide_same_stream_waits_v1(const launch_dependency_graph_v1&);
} // namespace cellerator::compiler::ir::realization::v1
