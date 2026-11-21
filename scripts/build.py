"""Build the Windows x64 model ports against the local zml_win fork."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = Path(os.environ.get("ZML_TEMP_ROOT") or ROOT / "_Temp")
BIN_DIR = ROOT / "validation-output" / "bin"
LOCK = json.loads((ROOT / "zml.lock.json").read_text(encoding="utf-8"))
TARGETS = ("//:zml_lib", "//:windows_consumer", "//:zml_lib_cpp", "//libraries/realesrgan:realesrgan_cli")
FORK_TARGETS = ("//examples/native_smoke", "//examples/mnist")


def check_engine(engine, env):
    command = ["git", "--no-optional-locks", "-C", str(engine)]
    head = subprocess.check_output([*command, "rev-parse", "HEAD"], text=True, env=env).strip()
    pin = LOCK["windows_revision"]
    if head != pin:
        subprocess.run([*command, "merge-base", "--is-ancestor", pin, head], check=True, env=env)
        subprocess.run([
            *command, "diff", "--exit-code", pin, head, "--", ".", ":(exclude)README.md", ":(exclude)docs/**",
        ], check=True, env=env)
    if subprocess.check_output([*command, "status", "--porcelain", "--untracked-files=no"], env=env):
        raise SystemExit("Tracked zml_win files differ from the selected revision")
    return head


@contextmanager
def scratch_directory():
    if not TEMP_ROOT.is_absolute() or TEMP_ROOT.drive.upper() == "C:" or TEMP_ROOT.resolve() != TEMP_ROOT:
        raise SystemExit(f"Expected a real temporary directory outside C: at {TEMP_ROOT}")
    TEMP_ROOT.mkdir(exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="run-", dir=TEMP_ROOT))

    def verify():
        if scratch.resolve() != scratch or scratch.parent.resolve() != TEMP_ROOT:
            raise OSError(f"Refusing access to a redirected scratch directory: {scratch}")

    def remove_readonly(remove, name, error):
        if not isinstance(error[1], PermissionError) or not Path(name).resolve().is_relative_to(scratch):
            raise error[1]
        os.chmod(name, stat.S_IWRITE)
        remove(name)

    verify()
    print(f"Scratch directory: {scratch}", flush=True)
    try:
        yield scratch
    finally:
        verify()
        shutil.rmtree(scratch, onerror=remove_readonly)
        print(f"Scratch directory removed: {scratch}", flush=True)


def find_bazel(tools, requested):
    if requested:
        return shutil.which(requested)
    downloads = tools / "bazelisk-home" / "downloads"
    metadata = downloads / "metadata" / "bazelbuild" / f"bazel-{LOCK['bazel']}-windows-x86_64"
    if metadata.is_file():
        digest = metadata.read_text(encoding="ascii").strip()
        if re.fullmatch(r"[a-f0-9]{64}", digest):
            pinned = downloads / "sha256" / digest / "bin" / "bazel.exe"
            if pinned.is_file():
                return str(pinned)
    bundled = tools / "bazelisk.exe"
    return str(bundled) if bundled.is_file() else shutil.which("bazelisk")


def run_environment(scratch, tools):
    temp = scratch / "tmp"
    temp.mkdir()
    zig_cache = (scratch / "zig").as_posix()
    env = os.environ.copy()
    env.update({
        "TEMP": temp.as_posix(),
        "TMP": temp.as_posix(),
        "TMPDIR": temp.as_posix(),
        "XDG_CACHE_HOME": (scratch / "xdg").as_posix(),
        "HF_HOME": (scratch / "huggingface").as_posix(),
        "TORCH_HOME": (scratch / "torch").as_posix(),
        "BAZELISK_HOME": (scratch / "bazelisk").as_posix(),
        "USE_BAZEL_VERSION": LOCK["bazel"],
        "RULES_ZIG_CACHE_PREFIX": zig_cache,
        "RULES_ZIG_CACHE_PREFIX_WINDOWS": zig_cache,
        "ZIG_GLOBAL_CACHE_DIR": zig_cache,
        "ZIG_LOCAL_CACHE_DIR": zig_cache,
        "PYTHONPYCACHEPREFIX": (scratch / "pycache").as_posix(),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    env.setdefault("BAZEL_SH", "C:/Program Files/Git/bin/bash.exe")
    if (tools / "empty.netrc").is_file():
        env.setdefault("NETRC", str(tools / "empty.netrc"))
    return env


def build(bazel, workspace, scratch, env, name, flags, targets):
    prefix = (scratch / f"{name}-").as_posix()
    startup = [
        bazel, "--batch", "--nosystem_rc", "--nohome_rc", "--noworkspace_rc",
        f"--bazelrc={(workspace / '.bazelrc').as_posix()}",
        f"--output_user_root={(scratch / 'bazel').as_posix()}",
        f"--output_base={(scratch / name).as_posix()}",
        f"--host_jvm_args=-Djava.io.tmpdir={env['TEMP']}",
    ]
    isolated = [
        "--lockfile_mode=off",
        f"--repository_cache={(scratch / 'downloads').as_posix()}",
        "--disk_cache=",
        f"--symlink_prefix={prefix}",
    ]
    for key in ("TEMP", "TMP", "TMPDIR"):
        isolated.extend(f"--{kind}_env={key}={env[key]}" for kind in ("repo", "action", "host_action"))
    for key in ("RULES_ZIG_CACHE_PREFIX", "RULES_ZIG_CACHE_PREFIX_WINDOWS"):
        isolated.append(f"--repo_env={key}={env[key]}")
    subprocess.run([*startup, "build", *flags, *isolated, *targets], cwd=workspace, env=env, check=True)
    bin_dir = (scratch / f"{name}-bin").resolve(strict=True)
    if not bin_dir.is_relative_to(scratch):
        raise OSError(f"Bazel binaries are outside the scratch directory: {bin_dir}")
    return bin_dir


def publish_binaries(bin_dir):
    for source in [*bin_dir.glob("*"), *bin_dir.glob("libraries/*/*")]:
        if source.suffix.lower() not in {".exe", ".dll", ".lib"} or not source.is_file():
            continue
        destination = BIN_DIR / source.relative_to(bin_dir)
        if destination.resolve() != destination:
            raise OSError(f"Refusing to write through a redirected path: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    print(f"Binaries: {BIN_DIR}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=ROOT.parent / "zml_win")
    parser.add_argument("--bazel", help="Bazel 9.3.0 or Bazelisk executable")
    parser.add_argument("--target", action="append", help="build only this Bazel target, can be repeated")
    parser.add_argument("--with-fork", action="store_true", help="also build the zml_win reference targets")
    parser.add_argument("bazel_args", nargs=argparse.REMAINDER, help="Bazel build arguments after --, optionally followed by --then COMMAND")
    args = parser.parse_args()
    extra = args.bazel_args[1:] if args.bazel_args[:1] == ["--"] else args.bazel_args
    test = []
    if "--then" in extra:
        delimiter = extra.index("--then")
        extra, test = extra[:delimiter], extra[delimiter + 1:]
        if not test:
            parser.error("--then requires a command")
    if "--" in extra:
        parser.error("Use one -- before the Bazel build arguments")
    if args.with_fork and not test:
        parser.error("--with-fork requires --then to use the reference binaries")
    engine = args.engine.resolve(strict=True)
    tools = engine / "bazel-win-tools"
    bazel = find_bazel(tools, args.bazel)
    if bazel is None:
        parser.error("Bazel was not found. Pass --bazel with its local path")

    with scratch_directory() as scratch:
        env = run_environment(scratch, tools)
        head = check_engine(engine, env)
        print(f"Windows ZML source pin {LOCK['windows_revision']}, checkout {head}", flush=True)
        if args.with_fork:
            fork_flags = [
                "--features=-layering_check", "--host_features=-layering_check",
                *(f"--@zml//platforms:{backend}=true" for backend in ("cpu", "cuda", "rocm", "vulkan")),
            ]
            fork_bin = build(bazel, engine, scratch, env, "fork", fork_flags, FORK_TARGETS)
            env["ZML_FORK_BIN_DIR"] = str(fork_bin)
        ports_flags = ["--config=windows_x64", f"--override_module=zml={engine.as_posix()}", *extra]
        bin_dir = build(bazel, ROOT, scratch, env, "ports", ports_flags, args.target or TARGETS)
        if test:
            env["ZML_BIN_DIR"] = str(bin_dir)
            env["ZML_TEST_DIR"] = str(scratch)
            command = [argument.replace("{bin}", str(bin_dir)) for argument in test]
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        else:
            publish_binaries(bin_dir)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(f"{error.cmd[0]} exited with status {error.returncode}", file=sys.stderr)
        sys.exit(error.returncode)
