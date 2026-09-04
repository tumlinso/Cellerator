#include <Cellerator/compiler/reflection/expose_reflection_and_inline_ir_diagnostics_v1.hh>
#include <sstream>
namespace cellerator::compiler::reflection::v1 {namespace{std::string esc(const std::string&s){std::string r;for(char c:s){if(c=='"'||c=='\\')r+='\\';r+=c;}return r;}}
std::string format_reflection_diagnostic_v1(const reflection_diagnostic_v1&d){std::ostringstream o;o<<(d.warning?"warning":"error")<<" R"<<static_cast<unsigned>(d.code)<<" "<<d.source<<": "<<d.message;if(!d.expected.empty()||!d.observed.empty())o<<" expected="<<d.expected<<" observed="<<d.observed;if(d.invalidations)o<<" invalidates="<<d.invalidations;return o.str();}
std::string serialize_reflection_diagnostics_v1(const std::vector<reflection_diagnostic_v1>&ds){std::ostringstream o;o<<"[";for(std::size_t i=0;i<ds.size();++i){if(i)o<<",";const auto&d=ds[i];o<<"{\"code\":"<<static_cast<unsigned>(d.code)<<",\"source\":\""<<esc(d.source)<<"\",\"message\":\""<<esc(d.message)<<"\",\"invalidations\":"<<d.invalidations<<",\"warning\":"<<(d.warning?"true":"false")<<"}";}o<<"]";return o.str();}
}
