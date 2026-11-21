#include "zml_lib.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

static int report(zml_lib_status status) {
    std::fprintf(stderr, "zml_lib status %u: %s\n", status, zml_lib_last_error());
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::fprintf(stderr, "Usage: zml_lib_cpp <weights.safetensors> <x2plus|x4plus|anime_6B> <cpu|cuda|rocm|vulkan>\n");
        return 2;
    }
    uint32_t variant = 0;
    if (std::strcmp(argv[2], "x2plus") == 0) variant = ZML_LIB_REALESRGAN_X2PLUS;
    if (std::strcmp(argv[2], "x4plus") == 0) variant = ZML_LIB_REALESRGAN_X4PLUS;
    if (std::strcmp(argv[2], "anime_6B") == 0) variant = ZML_LIB_REALESRGAN_X4PLUS_ANIME_6B;
    if (!variant) {
        std::fprintf(stderr, "Unsupported variant: %s\n", argv[2]);
        return 2;
    }
    const uint32_t scale = variant == ZML_LIB_REALESRGAN_X2PLUS ? 2 : 4;
    constexpr const char *backends[] = {"cpu", "cuda", "rocm", "vulkan"};
    uint32_t backend = 0;
    for (uint32_t i = 0; i < sizeof(backends) / sizeof(backends[0]); ++i) {
        if (std::strcmp(argv[3], backends[i]) == 0) backend = ZML_LIB_BACKEND_CPU + i;
    }
    if (!backend) {
        std::fprintf(stderr, "Unsupported backend: %s\n", argv[3]);
        return 2;
    }
    zml_lib_session *raw = nullptr;
    auto status = zml_lib_session_open(backend, &raw);
    if (status != ZML_LIB_OK) return report(status);
    const std::unique_ptr<zml_lib_session, decltype(&zml_lib_session_close)> session(raw, zml_lib_session_close);

    status = zml_lib_load_model(session.get(), variant, argv[1]);
    if (status != ZML_LIB_OK) return report(status);
    constexpr uint32_t width = 12;
    constexpr uint32_t height = 8;
    std::vector<float> input(3 * width * height, 0.25f);
    std::vector<float> output(3 * (width * scale) * (height * scale));
    for (int image = 0; image < 2; ++image) {
        input[0] = (image == 0) ? 0.0f : 1.0f;
        status = zml_lib_infer(session.get(), width, height, input.data(), input.size(), output.data(), output.size());
        if (status != ZML_LIB_OK) return report(status);
        std::printf("Image %d, first red sample: %g\n", image + 1, output[0]);
    }
    zml_lib_unload_model(session.get());
    return 0;
}
