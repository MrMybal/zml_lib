#ifndef ZML_LIB_H
#define ZML_LIB_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct zml_lib_session zml_lib_session;
typedef uint32_t zml_lib_status;

enum {
    ZML_LIB_OK = 0,
    ZML_LIB_INVALID_ARGUMENT = 1,
    ZML_LIB_NO_MODEL = 2,
    ZML_LIB_BACKEND_ERROR = 3,
    ZML_LIB_LOAD_ERROR = 4,
    ZML_LIB_INFERENCE_ERROR = 5,
    ZML_LIB_OUT_OF_MEMORY = 6
};

enum {
    ZML_LIB_BACKEND_AUTO = 0,
    ZML_LIB_BACKEND_CPU = 1,
    ZML_LIB_BACKEND_CUDA = 2,
    ZML_LIB_BACKEND_ROCM = 3,
    ZML_LIB_BACKEND_VULKAN = 4,
    ZML_LIB_BACKEND_METAL = 5
};

enum {
    ZML_LIB_REALESRGAN_X4PLUS = 1,
    ZML_LIB_REALESRGAN_X4PLUS_ANIME_6B = 2,
    ZML_LIB_REALESRGAN_X2PLUS = 3
};

/* Serialize calls on each session. On failure, *out_session is set to NULL. */
zml_lib_status zml_lib_session_open(uint32_t backend, zml_lib_session **out_session);

typedef struct {
    uint32_t backend;
    uint32_t device_count;
    uint32_t checked_output_shards;
    char pjrt_platform[32];
    char device[256];
} zml_lib_device_info;

/* Writes backend and device metadata into a caller-owned structure.
 * checked_output_shards counts placement checks on the last execution
 * before the output was copied to the host. */
zml_lib_status zml_lib_session_info(zml_lib_session *session, zml_lib_device_info *out);

/* Diagnostic operations on the selected backend. Loading the smoke model
 * replaces the previous model. The caller supplies two floats for its output. */
zml_lib_status zml_lib_add(zml_lib_session *session, float lhs, float rhs, float *output);
zml_lib_status zml_lib_load_smoke(zml_lib_session *session);
zml_lib_status zml_lib_infer_smoke(zml_lib_session *session, float output[2]);

/* Loads the MNIST reference SafeTensors. Inference borrows a 28x28 row-major
 * grayscale input with bytes in [0, 255] and writes a digit in [0, 9]. */
zml_lib_status zml_lib_load_mnist(zml_lib_session *session, const char *weights_path);
zml_lib_status zml_lib_infer_mnist(zml_lib_session *session, const uint8_t *pixels,
                                  size_t pixel_count, uint8_t *digit);

/* UTF-8, NUL-terminated local .safetensors path. A successful load replaces the
 * previous model. A failed load leaves it usable. The path is borrowed for
 * this call. Its file may be closed or moved after the call returns. */
zml_lib_status zml_lib_load_model(zml_lib_session *session, uint32_t variant, const char *weights_path);

/* Contiguous FP32 RGB, [1, 3, height, width], with input values in [0, 1].
 * The output is [1, 3, height*scale, width*scale], with raw, unclipped values.
 * The loaded variant specifies scale (2 for X2PLUS and 4 for X4 variants).
 * Counts are in float elements. The caller owns both buffers and keeps them
 * valid for the entire synchronous call. They may be reused upon return.
 * Output contents are unspecified after an inference error. */
zml_lib_status zml_lib_infer(zml_lib_session *session, uint32_t width, uint32_t height,
                             const float *input, size_t input_count,
                             float *output, size_t output_count);

/* Unload allows the session to be reused. Close also unloads a loaded model.
 * Both functions accept NULL. A closed handle becomes invalid immediately. */
void zml_lib_unload_model(zml_lib_session *session);
void zml_lib_session_close(zml_lib_session *session);

/* Borrowed, NUL-terminated error name for the last status-returning call on
 * this thread. Returns NULL after success. Do not free it. */
const char *zml_lib_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
