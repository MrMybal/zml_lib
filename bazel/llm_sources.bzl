"""Expose the LLM graphs from the selected ZML checkout without its terminal UI."""

def _llama_session(contents):
    # Decode appends the current token before writing it to the KV cache.
    cache_index = "self.token_index_buffers[all_tokens.items.len]"
    newline_lookup = 'self.tokenizer.tokenId("\\\\n") orelse return error.NoSuchToken'
    if cache_index not in contents or newline_lookup not in contents:
        fail("Unexpected Llama session in the pinned ZML checkout")
    contents = contents.replace(cache_index, "self.token_index_buffers[all_tokens.items.len - 1]")
    for token, error_name in [
        ("<|start_header_id|>", "MissingStartHeaderToken"),
        ("<|end_header_id|>", "MissingEndHeaderToken"),
        ("<|eot_id|>", "MissingEndOfTurnToken"),
    ]:
        lookup = 'self.tokenizer.tokenId("' + token + '") orelse return error.NoSuchToken'
        contents = contents.replace(lookup, lookup.replace("NoSuchToken", error_name))
    return contents.replace(
        newline_lookup,
        'self.tokenizer.tokenId("\\n") orelse self.tokenizer.tokenId("Ċ") orelse return error.MissingNewlineToken',
    )

def _llm_sources_impl(ctx):
    example = ctx.path(ctx.attr.module_file).dirname.get_child("examples").get_child("llm")
    # Materialize this small source tree so relative Zig imports use the overlay.
    ctx.file("models.zig", ctx.read(example.get_child("models.zig")), executable = False)
    for entry in example.get_child("models").readdir():
        if entry.basename.endswith(".zig"):
            ctx.file("models/" + entry.basename, ctx.read(entry), executable = False)
        else:
            for source in entry.readdir():
                destination = "models/" + entry.basename + "/" + source.basename
                contents = ctx.read(source)
                if destination == "models/llama/session.zig":
                    contents = _llama_session(contents)
                ctx.file(destination, contents, executable = False)
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
