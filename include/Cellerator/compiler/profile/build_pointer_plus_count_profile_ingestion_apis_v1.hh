#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef enum ce_profile_observation_kind_v1 { CE_PROFILE_RELATION_V1=1, CE_PROFILE_SUPPORT_V1=2, CE_PROFILE_VALUE_V1=3, CE_PROFILE_TRACE_V1=4 } ce_profile_observation_kind_v1;
typedef struct ce_profile_observation_span_v1 { const void* data; uint64_t count; uint32_t element_bytes; uint32_t kind; } ce_profile_observation_span_v1;
typedef struct ce_profile_ingestion_batch_v1 { ce_profile_observation_span_v1 relation, support, value, trace; } ce_profile_ingestion_batch_v1;
typedef int (*ce_profile_observation_sink_v1)(void*,const ce_profile_observation_span_v1*);
typedef struct ce_profile_ingestion_sink_v1 { void* context; ce_profile_observation_sink_v1 emit; } ce_profile_ingestion_sink_v1;
typedef enum ce_profile_ingestion_status_v1 { CE_PROFILE_INGEST_OK_V1=0, CE_PROFILE_INGEST_INVALID_V1=1, CE_PROFILE_INGEST_REJECTED_V1=2 } ce_profile_ingestion_status_v1;
ce_profile_ingestion_status_v1 ce_ingest_profile_observations_v1(const ce_profile_ingestion_batch_v1*,const ce_profile_ingestion_sink_v1*);
#ifdef __cplusplus
}
#endif
