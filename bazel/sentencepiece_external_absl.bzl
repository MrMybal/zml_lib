"""Use the Abseil dependency already selected by the pinned ZML build."""

def _sentencepiece_impl(ctx):
    ctx.download_and_extract(
        url = "https://github.com/google/sentencepiece/releases/download/v0.2.1/sentencepiece-0.2.1.tar.gz",
        sha256 = "8138cec27c2f2282f4a34d9a016e3374cd40e5c6e9cb335063db66a0a3b71fad",
        strip_prefix = "sentencepiece-0.2.1/sentencepiece",
    )
    # The vendored headers win quote-include lookup on Windows before ZML's
    # forwarding headers. Include the declared external Abseil headers directly.
    for source in ctx.path("src").readdir():
        if source.basename.endswith(".h") or source.basename.endswith(".cc"):
            content = ctx.read(source)
            fixed = content.replace('"third_party/absl/', '"absl/')
            if content != fixed:
                ctx.file("src/" + source.basename, fixed)
    ctx.file("BUILD.bazel", ctx.read(ctx.attr.engine_build))

_sentencepiece = repository_rule(
    implementation = _sentencepiece_impl,
    attrs = {"engine_build": attr.label(allow_single_file = True, mandatory = True)},
)

def _compat_impl(_ctx):
    _sentencepiece(
        name = "sentencepiece_external_absl",
        engine_build = Label("@zml//third_party/com_google_sentencepiece:sentencepiece.bazel"),
    )

sentencepiece_compat = module_extension(implementation = _compat_impl)
