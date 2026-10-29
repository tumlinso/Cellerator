#include <Cellerator/compiler/migration/reconcile_old_jbc_documentation_and_active_run_status_v1.hh>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
using namespace Cellerator::compiler::migration;
int main(int argc,char**argv){try{if(argc!=2)throw std::runtime_error("usage: test RECEIPT");std::set<std::string_view> classes;for(auto r:jbc_supersession_v1)if(!classes.insert(r.historical_class).second||!r.preserve_record||!r.additive_only)throw std::runtime_error("destructive supersession");if(classes.size()!=4||!preserves_history_v1())throw std::runtime_error("incomplete supersession");std::ifstream s(argv[1]);std::ostringstream o;o<<s.rdbuf();auto t=o.str();for(auto x:{"history_trace","revision 4118","CE-CCP1-I02-JBC-MIGRATION-MANIFEST","never deleted"})if(t.find(x)==std::string::npos)throw std::runtime_error("missing history evidence");std::cout<<"validated additive JBC supersession\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
