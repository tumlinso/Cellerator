#include "../../../src/compiler/tooling/cellerator/tooling_model.hh"
#include <cassert>
#include <string>
using namespace cellerator::compiler::tooling::v1;
int main(){auto all=complete_cellerator_syntax("field { rel",11);assert(all.size()==1&&all[0].spelling=="relation");auto malformed=complete_cellerator_syntax("field {\n  ",10);assert(malformed.size()>=12);auto native=complete_cellerator_syntax("nat",3);assert(native.size()==1&&native[0].category=="native-block");}
