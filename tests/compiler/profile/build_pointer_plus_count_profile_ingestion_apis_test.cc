#include <Cellerator/compiler/profile/build_pointer_plus_count_profile_ingestion_apis_v1.hh>
#include <cassert>
extern "C" ce_profile_ingestion_batch_v1 ce_test_c_profile_batch(void);
struct receipt { uint64_t calls=0,count=0; };
int accept(void* p,const ce_profile_observation_span_v1* s){auto&r=*static_cast<receipt*>(p);++r.calls;r.count+=s->count;return 1;}
int main(){auto batch=ce_test_c_profile_batch();receipt r{};ce_profile_ingestion_sink_v1 sink{&r,accept};assert(ce_ingest_profile_observations_v1(&batch,&sink)==CE_PROFILE_INGEST_OK_V1);assert(r.calls==4&&r.count==7);batch.value.data=nullptr;assert(ce_ingest_profile_observations_v1(&batch,&sink)==CE_PROFILE_INGEST_INVALID_V1);}
