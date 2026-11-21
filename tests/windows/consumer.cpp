#include "zml_lib.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>

struct Backend {
    const char *name;
    uint32_t id;
    const char *pjrt;
    const char *device;
};

static constexpr Backend backends[] = {
    {"cpu", ZML_LIB_BACKEND_CPU, "cpu", "cpu:0"},
    {"cuda", ZML_LIB_BACKEND_CUDA, "iree_cuda", "RTX 3090"},
    {"rocm", ZML_LIB_BACKEND_ROCM, "iree_rocm", "AMD Radeon"},
    {"vulkan", ZML_LIB_BACKEND_VULKAN, "iree_vulkan", "AMD Radeon"},
};

static bool check(zml_lib_status result, const char *step, zml_lib_status expected = ZML_LIB_OK) {
    if (result == expected) return true;
    const char *message = zml_lib_last_error();
    std::fprintf(stderr, "%s: expected=%u actual=%u error=%s\n", step, expected, result, message ? message : "");
    return false;
}

static bool verify_device(zml_lib_session *session, const Backend &backend, uint32_t outputs) {
    zml_lib_device_info info{};
    if (!check(zml_lib_session_info(session, &info), "session_info")) return false;
    if (info.backend != backend.id || std::strcmp(info.pjrt_platform, backend.pjrt) != 0 ||
        !std::strstr(info.device, backend.device) || info.device_count != 1 || info.checked_output_shards != outputs) {
        std::fprintf(stderr, "Unexpected placement: backend=%u pjrt=%s device=%s devices=%u checked=%u\n",
                     info.backend, info.pjrt_platform, info.device, info.device_count, info.checked_output_shards);
        return false;
    }
    std::printf("backend=%s pjrt=%s device=%s checked=%u\n",
                backend.name, info.pjrt_platform, info.device, info.checked_output_shards);
    return true;
}

static bool verify_smoke(zml_lib_session *session, const Backend &backend) {
    float output[2]{};
    if (!check(zml_lib_infer_smoke(session, output), "infer_smoke") ||
        !verify_device(session, backend, 1)) return false;
    std::printf("smoke=[%g,%g]\n", output[0], output[1]);
    return output[0] == 7 && output[1] == 0;
}

int main(int argc, char **argv) {
    bool expect_unavailable = argc == 3 && std::strcmp(argv[2], "--unavailable") == 0;
    if (argc != 4 && !expect_unavailable) {
        std::fprintf(stderr, "Usage: windows_consumer <backend> <mnist.safetensors> <images.idx3-ubyte>\n");
        return 2;
    }
    const Backend *selected = nullptr;
    for (const auto &backend : backends)
        if (std::strcmp(backend.name, argv[1]) == 0) selected = &backend;
    if (!selected) return 2;
    const Backend &backend = *selected;
    zml_lib_session *raw = nullptr;
    const auto opened = zml_lib_session_open(backend.id, &raw);
    const std::unique_ptr<zml_lib_session, decltype(&zml_lib_session_close)> session(raw, zml_lib_session_close);
    if (expect_unavailable) {
        const char *detail = zml_lib_last_error();
        if (!check(opened, "expected unavailable backend", ZML_LIB_BACKEND_ERROR) || raw || !detail || !*detail)
            return 1;
        std::printf("backend=%s unavailable status=%u error=%s\n", backend.name, opened, detail);
        return 0;
    }
    if (!check(opened, "session_open") || !raw || !verify_device(raw, backend, 0)) return 1;

    float sum = 0;
    if (!check(zml_lib_add(raw, 19, 23, &sum), "addition") || !verify_device(raw, backend, 1) || sum != 42)
        return 1;
    std::printf("19 + 23 = %g\n", sum);

    if (!check(zml_lib_load_smoke(raw), "load_smoke")) return 1;
    for (int i = 0; i < 10; ++i)
        if (!verify_smoke(raw, backend)) return 1;
    zml_lib_unload_model(raw);
    float discarded[2]{};
    if (!check(zml_lib_infer_smoke(raw, discarded), "unloaded_smoke", ZML_LIB_NO_MODEL) ||
        !check(zml_lib_load_smoke(raw), "reload_smoke") || !verify_smoke(raw, backend)) return 1;

    std::ifstream images(argv[3], std::ios::binary);
    std::array<unsigned char, 16> header{};
    images.read(reinterpret_cast<char *>(header.data()), header.size());
    if (!images || header[0] != 0 || header[1] != 0 || header[2] != 8 || header[3] != 3 ||
        header[11] != 28 || header[15] != 28) return 1;
    std::array<std::array<uint8_t, 784>, 3> samples{};
    for (auto &sample : samples)
        images.read(reinterpret_cast<char *>(sample.data()), sample.size());
    if (!images || !check(zml_lib_load_mnist(raw, argv[2]), "load_mnist")) return 1;

    constexpr uint8_t expected[] = {7, 2, 1};
    for (unsigned i = 0; i < 10; ++i) {
        const auto index = i % samples.size();
        uint8_t digit = 255;
        if (!check(zml_lib_infer_mnist(raw, samples[index].data(), samples[index].size(), &digit), "infer_mnist") ||
            !verify_device(raw, backend, 1) || digit != expected[index]) return 1;
        std::printf("mnist[%zu]=%u\n", index, static_cast<unsigned>(digit));
    }
    zml_lib_unload_model(raw);
    uint8_t digit = 255;
    if (!check(zml_lib_infer_mnist(raw, samples[0].data(), samples[0].size(), &digit), "unloaded_mnist", ZML_LIB_NO_MODEL) ||
        !check(zml_lib_load_mnist(raw, argv[2]), "reload_mnist") ||
        !check(zml_lib_infer_mnist(raw, samples[0].data(), samples[0].size(), &digit), "infer_reloaded_mnist") ||
        !verify_device(raw, backend, 1) || digit != expected[0]) return 1;
    std::printf("mnist_reloaded=%u\n", static_cast<unsigned>(digit));
    return 0;
}
