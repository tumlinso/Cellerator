#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cellerator_compiler_session_v1 cellerator_compiler_session_v1;
typedef void (*cellerator_diagnostic_callback_v1)(int severity, const char* message,
    void* user_data);
typedef int (*cellerator_cancel_callback_v1)(void* user_data);

typedef struct cellerator_compiler_config_v1 {
    uint32_t struct_size;
    const char* target;
    const char* toolchain;
    const char* profile;
    cellerator_diagnostic_callback_v1 diagnostic;
    cellerator_cancel_callback_v1 cancelled;
    void* user_data;
} cellerator_compiler_config_v1;

typedef struct cellerator_compiler_output_v1 {
    uint32_t struct_size;
    const char* object_identity;
    size_t source_count;
} cellerator_compiler_output_v1;

cellerator_compiler_session_v1* cellerator_compiler_session_create_v1(
    const cellerator_compiler_config_v1* config);
void cellerator_compiler_session_destroy_v1(cellerator_compiler_session_v1* session);
int cellerator_compiler_session_add_source_buffer_v1(cellerator_compiler_session_v1* session,
    const char* name, const char* data, size_t size);
int cellerator_compiler_session_add_source_file_v1(cellerator_compiler_session_v1* session,
    const char* path);
int cellerator_compiler_session_compile_v1(cellerator_compiler_session_v1* session,
    cellerator_compiler_output_v1* output);

#ifdef __cplusplus
}
#endif
