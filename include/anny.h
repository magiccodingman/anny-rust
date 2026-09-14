#ifndef ANNY_RUST_H
#define ANNY_RUST_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* ABI v1. Build separately for each OS/architecture. All functions are cdecl.
 * 0=success, 1=error, 2=caught Rust panic. Strings are UTF-8 NUL terminated.
 * Null config/parameter strings mean defaults. Never pass invalid/dangling pointers.
 * A model is immutable after creation; independent output handles own their arrays.
 * Tensor buffers are row-major f64, including exact-integer indices; kind identifies
 * semantic type (0=float,1=index,2=bool). Positions are Z-up meters. Matrix order is
 * row-major. Views are borrowed READ-ONLY until the respective handle is freed.
 * Never free library memory with free()/delete/Marshal.FreeHGlobal.
 * Concurrent evaluation is allowed; destruction must be synchronized by the caller.
 */
typedef struct AnnyModel AnnyModel;
typedef struct AnnyOutput AnnyOutput;
typedef struct { const double *data; size_t len; const size_t *shape; size_t rank; uint32_t kind; } AnnyTensorView;
uint32_t anny_abi_version(void);
const char *anny_last_error(void); /* borrowed, same-thread next-call lifetime */
int32_t anny_model_from_bytes(const uint8_t *bytes,size_t len,const char *config_json,AnnyModel **out);
int32_t anny_model_load(const char *path,const char *config_json,AnnyModel **out);
int32_t anny_model_build(const char *assets,const char *config_json,AnnyModel **out);
void anny_model_free(AnnyModel *model);
int32_t anny_model_evaluate(const AnnyModel *model,const char *parameters_json,AnnyOutput **out);
void anny_output_free(AnnyOutput *output);
int32_t anny_model_tensor(const AnnyModel *model,const char *name,AnnyTensorView *out);
int32_t anny_output_tensor(const AnnyOutput *output,const char *name,AnnyTensorView *out);
int32_t anny_model_describe(const AnnyModel *model,char **out);
void anny_string_free(char *s);
/* Additive ABI-1 export API; GLB has standard Y-up coordinates and f32 attributes. */
typedef struct AnnyBytes AnnyBytes;
int32_t anny_model_export_glb(const AnnyModel *model,const char *parameters_json,const char *options_json,AnnyBytes **out);
const uint8_t *anny_bytes_data(const AnnyBytes *bytes);
size_t anny_bytes_len(const AnnyBytes *bytes);
void anny_bytes_free(AnnyBytes *bytes);
#ifdef __cplusplus
}
#endif
#endif
