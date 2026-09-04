#include <Cellerator/compiler/migration/define_cellerator_ruleset_export_consumed_by_cellshard_v1.hh>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace Cellerator::compiler::migration;
int main(int argc,char**argv){try{if(argc!=2)throw std::runtime_error("usage: test HEADER");ruleset_export_v1 r{1,sizeof(r),1,2,3,4,5,6,7,8,9};if(!valid(r))throw std::runtime_error("valid export rejected");r.structure_generation=0;if(valid(r))throw std::runtime_error("unstamped export accepted");std::ifstream s(argv[1]);std::ostringstream o;o<<s.rdbuf();auto t=o.str();for(auto x:{"CellShard/","file_path","object_key","device_ordinal","lease_token","void *"})if(t.find(x)!=std::string::npos)throw std::runtime_error("concrete dependency in ruleset export");std::cout<<"validated standalone immutable ruleset export\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
