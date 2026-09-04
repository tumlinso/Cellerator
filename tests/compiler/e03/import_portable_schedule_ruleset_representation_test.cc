#include <Cellerator/compiler/program/import_portable_schedule_ruleset_representation_v1.hh>
#include <cassert>
using namespace Cellerator::compiler::composition;
int main(){portable_schedule_v1 s{{"load","apply"},{"atom.a"},{"merge(sum)"},{"canonical(gene)"},replay_mode_v1::compatible};std::string e;auto id=portable_schedule_identity_v1(s);assert(id&&validate_portable_schedule_v1(s,&e)&&portable_schedule_identity_v1(s)==id);auto path=s;path.atom_requirements={"/tmp/a"};assert(!validate_portable_schedule_v1(path,&e)&&portable_schedule_identity_v1(path)==0);auto addr=s;addr.partial_tree={"buffer@0x42"};assert(!validate_portable_schedule_v1(addr,&e));}
