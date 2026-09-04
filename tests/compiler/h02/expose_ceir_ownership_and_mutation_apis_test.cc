#include <Cellerator/sdk/expose_ceir_ownership_and_mutation_apis_v1.hh>
#include <cassert>
namespace ca=cellerator::compiler::api::v1;
int main(){for(auto level:{ca::ceir_level_v1::semantic,ca::ceir_level_v1::planning,ca::ceir_level_v1::realization}){auto m=ca::parse_ceir_v1(level,"op %1");assert(ca::validate_ceir_v1(m,ca::ceir_validation_v1::checked));auto b=ca::clone_ceir_v1(m);b.set_text("edited %1");b.add_provenance("external-editor");auto e=b.freeze();assert(ca::print_ceir_v1(e)=="edited %1");assert(e->provenance.size()==2);assert(!ca::serialize_ceir_v1(e).empty());}}
