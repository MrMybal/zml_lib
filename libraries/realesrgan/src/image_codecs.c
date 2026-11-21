#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_NO_STDIO
#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"

#define STBI_WRITE_NO_STDIO
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb_image_write.h"

#include <stdint.h>
#include <stdio.h>

/* Reserve the protocol pipe before loading a runtime that may print to stdout. */
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <io.h>

intptr_t zml_lib_cli_reserve_stdout(void) {
    HANDLE stdout_copy;
    HANDLE process = GetCurrentProcess();
    if (!DuplicateHandle(process, GetStdHandle(STD_OUTPUT_HANDLE), process, &stdout_copy,
                         0, FALSE, DUPLICATE_SAME_ACCESS))
        return -1;
    if (_dup2(_fileno(stderr), _fileno(stdout)) != 0 ||
        !SetStdHandle(STD_OUTPUT_HANDLE, GetStdHandle(STD_ERROR_HANDLE))) {
        CloseHandle(stdout_copy);
        return -1;
    }
    return (intptr_t)stdout_copy;
}
#else
#include <fcntl.h>
#include <unistd.h>

intptr_t zml_lib_cli_reserve_stdout(void) {
    const int stdout_copy = fcntl(STDOUT_FILENO, F_DUPFD_CLOEXEC, 3);
    if (stdout_copy < 0)
        return -1;
    if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
        close(stdout_copy);
        return -1;
    }
    return stdout_copy;
}
#endif
