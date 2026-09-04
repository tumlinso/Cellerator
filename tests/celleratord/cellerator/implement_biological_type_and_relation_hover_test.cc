#include "../../../src/compiler/tooling/cellerator/tooling_model.hh"
#include <cassert>
using namespace cellerator::compiler::tooling::v1;
int main(){auto h=describe_biological_relation("relation learned transpose");assert(h.domain=="gene"&&h.orientation=="transpose"&&h.mutability.find("learned")!=h.mutability.npos);assert(!h.source_axis.empty()&&!h.destination_axis.empty()&&!h.numeric_tuple.empty()&&!h.source_link.empty());}
