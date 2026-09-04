#include <Cellerator/compiler/composition/import_basis_manifest_semantics_v1.hh>
#include <iomanip>
#include <sstream>
namespace Cellerator::compiler::composition {
std::string print_basis_manifest_v1(const basis_manifest_v1&m){std::ostringstream o;o<<std::setprecision(17)<<m.id<<'\n'<<m.evidence_fingerprint<<'\n'<<m.budget_bytes<<' '<<m.evidence_generation<<' '<<m.valid<<' '<<m.objective_vector.size()<<' '<<m.members.size()<<'\n';for(double x:m.objective_vector)o<<x<<' ';o<<'\n';for(const auto&x:m.members)o<<x.atom<<'\t'<<x.production<<'\t'<<x.membership<<'\t'<<x.redundancy<<'\n';return o.str();}
std::optional<basis_manifest_v1> parse_basis_manifest_v1(const std::string&s){std::istringstream i(s);basis_manifest_v1 m;std::size_t no=0,nm=0;if(!std::getline(i,m.id)||!std::getline(i,m.evidence_fingerprint)||!(i>>m.budget_bytes>>m.evidence_generation>>m.valid>>no>>nm)||m.id.empty()||m.evidence_fingerprint.empty())return std::nullopt;m.objective_vector.resize(no);for(auto&x:m.objective_vector)if(!(i>>x))return std::nullopt;i.ignore(1024,'\n');for(std::size_t n=0;n<nm;++n){basis_member_v1 x;if(!std::getline(i,x.atom,'\t')||!std::getline(i,x.production,'\t')||!std::getline(i,x.membership,'\t')||!(i>>x.redundancy))return std::nullopt;i.ignore(1024,'\n');m.members.push_back(std::move(x));}return m;}
}
