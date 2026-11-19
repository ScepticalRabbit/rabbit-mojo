"""Compile a CPython extension and vendor its three required Mojo runtime libraries."""

from importlib.metadata import distribution
import os
from pathlib import Path
import shutil
import subprocess
import sys
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

RUNTIMES = (
    "libKGENCompilerRTShared.so",
    "libMSupportGlobals.so",
    "libAsyncRTRuntimeGlobals.so",
)


class MojoBuild(build_ext):
    def build_extension(self, ext):
        if sys.platform != "linux" or os.uname().machine != "x86_64":
            raise RuntimeError("rabbit-mojo currently supports Linux x86-64 only")
        compiler = os.environ.get("MOJO") or shutil.which("mojo")
        if not compiler:
            candidate = Path(sys.executable).parent / "mojo"
            if candidate.exists():
                compiler = str(candidate)
        if not compiler:
            raise RuntimeError(
                "Mojo compiler missing: uv pip install 'mojo==1.2.0', or set MOJO"
            )
        patcher = shutil.which("patchelf")
        if not patcher:
            candidate = Path(sys.executable).parent / "patchelf"
            if candidate.exists():
                patcher = str(candidate)
        if not patcher:
            raise RuntimeError("patchelf missing: uv pip install patchelf")
        version = subprocess.check_output([compiler, "--version"], text=True)
        if "Mojo 1.2.0 " not in version:
            raise RuntimeError(
                f"Verified compiler is Mojo 1.2.0; found {version.strip()}"
            )
        target = Path(self.get_ext_fullpath(ext.name))
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                compiler,
                "build",
                "--emit",
                "shared-lib",
                "-O3",
                "--target-cpu",
                "x86-64",
                ext.sources[0],
                "-o",
                str(target),
            ],
            check=True,
        )
        runtime_directory = Path(
            distribution("mojo-compiler").locate_file("modular/lib")
        )
        destination = target.parent / ".libs"
        destination.mkdir(exist_ok=True)
        for name in RUNTIMES:
            library = destination / name
            shutil.copy2(runtime_directory / name, library)
            subprocess.run(
                [patcher, "--set-rpath", "$ORIGIN", str(library)], check=True
            )
        subprocess.run(
            [patcher, "--set-rpath", "$ORIGIN/.libs", str(target)], check=True
        )

    def copy_extensions_to_source(self):
        super().copy_extensions_to_source()
        source = Path(self.build_lib) / "rabbit_mojo" / ".libs"
        destination = (
            Path(self.get_finalized_command("build_py").get_package_dir("rabbit_mojo"))
            / ".libs"
        )
        shutil.copytree(source, destination, dirs_exist_ok=True)


setup(
    ext_modules=[
        Extension("rabbit_mojo._native", ["src/rabbit_mojo/_mojo/bindings.mojo"])
    ],
    cmdclass={"build_ext": MojoBuild},
)
