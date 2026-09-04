#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CELLERATOR_HOST_BINDING_ABI_VERSION_V1 1u

typedef enum cellerator_host_status_v1 {
    CELLERATOR_HOST_SUCCESS_V1 = 0,
    CELLERATOR_HOST_INVALID_ARGUMENT_V1 = 1,
    CELLERATOR_HOST_UNSUPPORTED_ABI_V1 = 2,
    CELLERATOR_HOST_INVALID_OPERAND_V1 = 3,
    CELLERATOR_HOST_INSUFFICIENT_WORKSPACE_V1 = 4,
    CELLERATOR_HOST_STAGE_FAILED_V1 = 5
} cellerator_host_status_v1;

typedef enum cellerator_host_operand_kind_v1 {
    CELLERATOR_HOST_INPUT_V1 = 1,
    CELLERATOR_HOST_OUTPUT_V1 = 2,
    CELLERATOR_HOST_MUTABLE_VALUE_V1 = 3
} cellerator_host_operand_kind_v1;

typedef struct cellerator_host_operand_v1 {
    void* data;
    size_t bytes;
    uint32_t element_size;
    uint32_t kind;
} cellerator_host_operand_v1;

typedef struct cellerator_host_constant_v1 {
    const void* data;
    size_t bytes;
} cellerator_host_constant_v1;

struct cellerator_host_binding_v1;
typedef cellerator_host_status_v1 (*cellerator_host_stage_v1)(
    void* context, const struct cellerator_host_binding_v1* binding);

typedef struct cellerator_host_prepared_stage_v1 {
    cellerator_host_stage_v1 run;
    void* context;
} cellerator_host_prepared_stage_v1;

typedef struct cellerator_host_binding_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    const cellerator_host_operand_v1* operands;
    uint32_t operand_count;
    const cellerator_host_constant_v1* constants;
    uint32_t constant_count;
    void* workspace;
    size_t workspace_bytes;
    size_t required_workspace_bytes;
    const cellerator_host_prepared_stage_v1* stages;
    uint32_t stage_count;
} cellerator_host_binding_v1;

cellerator_host_status_v1 cellerator_host_execute_v1(
    const cellerator_host_binding_v1* binding);

#ifdef __cplusplus
}
#endif
