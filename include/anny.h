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
typedef struct AnnySession AnnySession;
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
/* Reusable pose sessions: the phenotype/local-change/facial coefficients and the
 * rest model are computed once in anny_session_new, so each anny_session_update
 * costs only the pose-dependent half. Use them for animation, editor sliders and
 * any repeated re-posing of one parameter set, where anny_model_evaluate would
 * redo the fixed half every frame. A session owns a reference to its model, so the
 * model handle may be freed first. Views from anny_session_tensor are invalidated
 * by the next update of that session. */
int32_t anny_session_new(const AnnyModel *model,const char *parameters_json,AnnySession **out);
int32_t anny_session_update(AnnySession *session,const char *pose_json);
int32_t anny_session_tensor(const AnnySession *session,const char *name,AnnyTensorView *out);
int32_t anny_session_coefficients(const AnnySession *session,AnnyTensorView *out);
void anny_session_free(AnnySession *session);
void anny_string_free(char *s);
/* Additive ABI-1 export API; GLB has standard Y-up coordinates and f32 attributes. */
typedef struct AnnyBytes AnnyBytes;
int32_t anny_model_export_glb(const AnnyModel *model,const char *parameters_json,const char *options_json,AnnyBytes **out);
const uint8_t *anny_bytes_data(const AnnyBytes *bytes);
size_t anny_bytes_len(const AnnyBytes *bytes);
void anny_bytes_free(AnnyBytes *bytes);
/* Secondary operations use docs/AUTHORING.md request schemas. Query/transfer
 * text is freed with anny_string_free; transformed models are independently owned. */
int32_t anny_model_build_cached(const char *assets,const char *config_json,const char *cache_directory,AnnyModel **out);
int32_t anny_model_query(const AnnyModel *model,const char *request_json,char **out);
int32_t anny_model_transform(const AnnyModel *model,const char *operations_json,AnnyModel **out);
int32_t anny_model_prepared_bytes(const AnnyModel *model,AnnyBytes **out);
int32_t anny_model_transfer_pose(const AnnyModel *source,const AnnyModel *target,const char *parameters_json,const char *mode,char **out);
/* Additive single-precision API: separate handles/views preserve every ABI-1
 * double API. Model conversion is one-time; evaluation arithmetic is f32.
 * Static integer-index views are exactly represented floats (checked on import).
 * Ownership/thread rules are the same as the double APIs above. */
typedef struct AnnyModelF32 AnnyModelF32;
typedef struct AnnyOutputF32 AnnyOutputF32;
typedef struct AnnySessionF32 AnnySessionF32;
typedef struct { const float *data; size_t len; const size_t *shape; size_t rank; uint32_t kind; } AnnyTensorViewF32;
int32_t anny_model_to_f32(const AnnyModel *source,AnnyModelF32 **out);
int32_t anny_model_f32_from_bytes(const uint8_t *bytes,size_t len,const char *config_json,AnnyModelF32 **out);
int32_t anny_model_f32_evaluate(const AnnyModelF32 *model,const char *parameters_json,AnnyOutputF32 **out);
int32_t anny_model_f32_tensor(const AnnyModelF32 *model,const char *name,AnnyTensorViewF32 *out);
int32_t anny_output_f32_tensor(const AnnyOutputF32 *output,const char *name,AnnyTensorViewF32 *out);
int32_t anny_model_f32_prepared_bytes(const AnnyModelF32 *model,AnnyBytes **out);
int32_t anny_session_f32_new(const AnnyModelF32 *model,const char *parameters_json,AnnySessionF32 **out);
int32_t anny_session_f32_update(AnnySessionF32 *session,const char *pose_json);
int32_t anny_session_f32_tensor(const AnnySessionF32 *session,const char *name,AnnyTensorViewF32 *out);
int32_t anny_session_f32_coefficients(const AnnySessionF32 *session,AnnyTensorViewF32 *out);
void anny_session_f32_free(AnnySessionF32 *session);
void anny_model_f32_free(AnnyModelF32 *model);
void anny_output_f32_free(AnnyOutputF32 *output);
/* Standalone base-glTF authoring, no model handle required. Inputs are borrowed
 * for the call. On error outputs are null; error text uses anny_last_error(). */
int32_t anny_gltf_edit(const uint8_t *bytes, size_t len,
    const char *operations_json, AnnyBytes **out);
int32_t anny_gltf_query(const uint8_t *bytes, size_t len,
    const char *request_json, char **out);

#ifdef __cplusplus
}
#endif
#endif
