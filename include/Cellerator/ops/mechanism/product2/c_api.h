#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef enum ce_product2_status {
 CE_PRODUCT2_SUCCESS=0, CE_PRODUCT2_INVALID_ARGUMENT=1, CE_PRODUCT2_INVALID_INDEX=2,
 CE_PRODUCT2_STALE_GENERATION=3, CE_PRODUCT2_ALIAS=4, CE_PRODUCT2_OVERFLOW=5,
 CE_PRODUCT2_BACKEND_ERROR=6, CE_PRODUCT2_UNAVAILABLE=7
} ce_product2_status;
typedef struct ce_product2_context ce_product2_context;
typedef struct ce_product2_cuda_context ce_product2_cuda_context;
typedef struct ce_product2_binding {
 const float *x; uint64_t x_count;
 const float *k; uint64_t k_count;
 float *y; uint64_t y_count;
 const float *g; uint64_t g_count;
 const float *dx; uint64_t dx_count;
 const float *dk; uint64_t dk_count;
 float *dy; uint64_t dy_count;
 float *gx; uint64_t gx_count;
 float *gk; uint64_t gk_count;
 uint64_t expected_structure_generation;
 uint64_t current_structure_generation;
 uint64_t expected_value_generation;
 uint64_t current_value_generation;
 uint64_t expected_parameter_generation;
 uint64_t current_parameter_generation;
} ce_product2_binding;
/* Copies topology. Every count is an accessible element extent, not a byte size.
   Generations and borrowed buffers remain caller-owned and externally synchronized.
   No caller output is modified on admission failure. */
ce_product2_status ce_product2_create(uint64_t n,uint64_t m,const int64_t *a,uint64_t a_count,const int64_t *b,uint64_t b_count,uint64_t structure_generation,ce_product2_context **out);
void ce_product2_destroy(ce_product2_context *context);
ce_product2_status ce_product2_forward(const ce_product2_context*,const ce_product2_binding*);
ce_product2_status ce_product2_vjp(const ce_product2_context*,const ce_product2_binding*);
ce_product2_status ce_product2_jvp(const ce_product2_context*,const ce_product2_binding*);
ce_product2_status ce_product2_cuda_create(uint64_t n,uint64_t m,const int64_t *a,uint64_t a_count,const int64_t *b,uint64_t b_count,uint64_t structure_generation,void *stream,ce_product2_cuda_context **out);
void ce_product2_cuda_destroy(ce_product2_cuda_context*);
/* Pure admission: operation 0 forward, 1 VJP, 2 JVP. */
/* Pure exact copied-topology comparison for typed prepared-stage binding. */
ce_product2_status ce_product2_cuda_matches(const ce_product2_cuda_context*,uint64_t n,uint64_t m,const int64_t *a,uint64_t a_count,const int64_t *b,uint64_t b_count,uint64_t structure_generation);
ce_product2_status ce_product2_cuda_admit(const ce_product2_cuda_context*,const ce_product2_binding*,void *stream,int operation);
ce_product2_status ce_product2_cuda_forward(const ce_product2_cuda_context*,const ce_product2_binding*,void *stream);
ce_product2_status ce_product2_cuda_vjp(const ce_product2_cuda_context*,const ce_product2_binding*,void *stream);
ce_product2_status ce_product2_cuda_jvp(const ce_product2_cuda_context*,const ce_product2_binding*,void *stream);
#ifdef __cplusplus
}
#endif
