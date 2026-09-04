#include <Cellerator/compiler/migration/define_cellerator_ownership_of_basis_selection_v1.hh>
#include <iostream>
#include <stdexcept>
using namespace Cellerator::compiler::migration;
int main(){try{representative_profile_basis_input_v1 in{1,2,3,4};portable_ruleset_basis_output_v1 selected{5,6,1,2,basis_outcome_v1::selected};if(!traceable(in,selected))throw std::runtime_error("selected basis not traceable");auto none=selected;none.basis_identity=0;none.outcome=basis_outcome_v1::no_basis;if(!traceable(in,none))throw std::runtime_error("no-basis outcome rejected");none.input_profile_generation=9;if(traceable(in,none))throw std::runtime_error("stale profile accepted");if(total({1,2,3,4,5,6})!=21)throw std::runtime_error("incomplete cost");std::cout<<"validated representative-profile basis ownership\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
