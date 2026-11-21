"""Expose the LLM graphs from the selected ZML checkout without its terminal UI."""

def _llm_sources_impl(ctx):
    example = ctx.path(ctx.attr.module_file).dirname.get_child("examples").get_child("llm")
    ctx.symlink(example.get_child("models.zig"), "models.zig")
    ctx.symlink(example.get_child("models"), "models")
    ctx.file("BUILD.bazel", """
load("@rules_zig//zig:defs.bzl", "zig_library")

zig_library(
    name = "models",
    main = "models.zig",
    srcs = glob(["models/**/*.zig"], exclude = ["models/*_tests.zig"]),
    import_name = "zml_llm_models",
    visibility = ["//visibility:public"],
    deps = ["@zml//zml"],
)
""")

_llm_sources = repository_rule(
    implementation = _llm_sources_impl,
    attrs = {"module_file": attr.label(allow_single_file = True, mandatory = True)},
)

def _extension_impl(_ctx):
    _llm_sources(name = "zml_llm_sources", module_file = Label("@zml//:MODULE.bazel"))

llm_sources = module_extension(implementation = _extension_impl)
