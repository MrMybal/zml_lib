def tokenizer_copts():
    # The pinned standalone translator predates Clang 23 AVX512-BMM builtins.
    # Neither tokenizer uses these intrinsics.
    return select({
        "@platforms//os:windows": ["-D_AVX512BMMINTRIN_H", "-D__BMMVLINTRIN_H"],
        "//conditions:default": [],
    })
