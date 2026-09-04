#include <Cellerator/compiler/program/define_the_concrete_cellshard_materialization_request_se_v1.hh>
#include <cassert>
using namespace Cellerator::compiler::composition;
int main(){portable_schedule_v1 s{{"apply"},{"atom.a"},{"sum"},{"canonical"},replay_mode_v1::exact};auto r=make_cellshard_materialization_request_v1(s,2,4,4096,{"sm70"});assert(r&&r->schedule_identity==portable_schedule_identity_v1(s)&&r->delivery_contract=="opaque-cellerator-execution-image-v1");assert(!make_cellshard_materialization_request_v1(s,0,4,4096,{"sm70"}));assert(!make_cellshard_materialization_request_v1(s,2,4,0,{"sm70"}));}
