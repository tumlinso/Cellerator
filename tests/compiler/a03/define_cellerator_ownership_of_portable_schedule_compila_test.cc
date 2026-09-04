#include <Cellerator/compiler/migration/define_cellerator_ownership_of_portable_schedule_compila_v1.hh>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace Cellerator::compiler::migration;
int main(int argc,char**argv){try{if(argc!=2)throw std::runtime_error("usage: test HEADER");std::ifstream s(argv[1]);std::ostringstream o;o<<s.rdbuf();auto t=o.str();for(auto x:{"path","pointer","lease","device_ordinal","route","CellShard"})if(t.find(x)!=std::string::npos)throw std::runtime_error("concrete field in portable identity");portable_schedule_identity_v1 v{1,2,3,4,5,6,7,8};if(!valid(v))throw std::runtime_error("valid schedule rejected");const portable_schedule_identity_v1 missing[]={{0,2,3,4,5,6,7,8},{1,0,3,4,5,6,7,8},{1,2,0,4,5,6,7,8},{1,2,3,0,5,6,7,8},{1,2,3,4,0,6,7,8},{1,2,3,4,5,0,7,8},{1,2,3,4,5,6,0,8},{1,2,3,4,5,6,7,0}};for(auto p:missing)if(valid(p))throw std::runtime_error("partial identity accepted");std::cout<<"validated portable schedule identity\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
