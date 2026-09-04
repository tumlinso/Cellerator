#include <Cellerator/compiler/migration/define_cellerator_ownership_of_superatom_promotion_v1.hh>
#include <iostream>
#include <stdexcept>
using namespace Cellerator::compiler::migration;
int main(){try{superatom_promotion_evidence_v1 fast{1,2,100,80,3,true};if(evaluate_superatom(fast)!=superatom_disposition_v1::promoted||is_storage_shard(fast))throw std::runtime_error("valid promotion failed");auto slow=fast;slow.composed_total_ns=100;if(evaluate_superatom(slow)!=superatom_disposition_v1::evaluated_not_promoted)throw std::runtime_error("non-promotion invalid");auto invalid=fast;invalid.deconstruction_digest=0;if(evaluate_superatom(invalid)!=superatom_disposition_v1::invalid)throw std::runtime_error("missing deconstruction accepted");std::cout<<"validated superatom promotion and non-promotion\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
