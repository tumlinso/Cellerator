#include "tooling_model.hh"
namespace cellerator::compiler::tooling::v1 {candidate_explanation explain_candidate(std::string_view k,bool forced){return {forced?300.:120.,20.,std::string(k),k=="cached"?"cached/current":"current","plus-minus 5%","memory <= 1MiB","structure x100","reference=300ns","dominates latency and bytes",forced?"forced by user":"none","reference"};}}
