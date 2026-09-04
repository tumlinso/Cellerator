#include "tooling_model.hh"
namespace cellerator::compiler::tooling::v1 {generation_view query_generations(std::string_view s){generation_view g;if(s.find("value")!=s.npos){++g.value;g.stale_artifacts={"value-binding"};}if(s.find("structure")!=s.npos){++g.structure;++g.support;++g.order;g.stale_artifacts={"candidate-cache","prepared-plan","projection"};}return g;}}
