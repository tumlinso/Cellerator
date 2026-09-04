#include <Cellerator/compiler/migration/define_cellerator_ownership_of_global_operation_program_v1.hh>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
using namespace Cellerator::compiler::migration;
int main(int argc,char**argv){try{if(argc!=2)throw std::runtime_error("usage: test HEADER");std::ifstream s(argv[1]);std::ostringstream o;o<<s.rdbuf();auto text=o.str();for(auto token:{"CellShard","file_path","object_key","lease_token","device_ordinal","void *"})if(text.find(token)!=std::string::npos)throw std::runtime_error("storage API in program IR");for(auto kind:{program_ir_entity_v1::field,program_ir_entity_v1::operation,program_ir_entity_v1::atom_flow,program_ir_entity_v1::profile_family})if(!valid({{1,2},{3,4},{5,6},kind,{},7}))throw std::runtime_error("valid entity rejected");if(valid({{1,2},{0,4},{5,6},program_ir_entity_v1::field,{},7}))throw std::runtime_error("missing domain accepted");std::cout<<"validated storage-free global program IR\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
