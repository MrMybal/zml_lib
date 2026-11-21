#ifndef ZML_FACES_H
#define ZML_FACES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct zml_faces_session zml_faces_session;

enum zml_faces_backend {
    ZML_FACES_AUTO = 0, ZML_FACES_CPU = 1, ZML_FACES_CUDA = 2,
    ZML_FACES_ROCM = 3, ZML_FACES_VULKAN = 4
};

enum zml_faces_model {
    ZML_FACES_RETINAFACE = 1, ZML_FACES_PARSENET = 2, ZML_FACES_CODEFORMER = 3
};

typedef struct zml_faces_device_info {
    uint32_t backend;
    uint32_t output_shards;
    char pjrt_platform[32];
    char device[256];
} zml_faces_device_info;

/* Nonzero status means failure. Error name lives until the next call on this thread. */
uint32_t zml_faces_open(uint32_t backend, zml_faces_session **out);
void zml_faces_close(zml_faces_session *session);
const char *zml_faces_last_error(void);
uint32_t zml_faces_load(zml_faces_session *session, uint32_t model, const char *weights_utf8);
uint32_t zml_faces_info(const zml_faces_session *session, zml_faces_device_info *out);

/* Buffers are flat contiguous FP32, borrowed until the call returns. Serialize calls per session.
 * RetinaFace: BGR NCHW minus [104,117,123]. Output per anchor: box[4], probability[2], landmarks[10].
 * ParseNet: RGB NCHW [-1,1] at 512x512. Output: nineteen logit planes then three RGB planes.
 * CodeFormer: same image input, fidelity in [0,1]. Output: three RGB planes then 256x1024 logits.
 */
uint32_t zml_faces_infer(zml_faces_session *session, uint32_t model, uint32_t width, uint32_t height,
                         float fidelity, const float *input, size_t input_count,
                         float *output, size_t output_count);

/* RetinaFace decode and NMS. Each returned detection is box[4], score, landmarks[10]. */
uint32_t zml_faces_decode(uint32_t width, uint32_t height, const float *raw, size_t raw_count,
                         float *faces, size_t face_capacity_floats, size_t *face_count);

#ifdef __cplusplus
}
#endif
#endif
