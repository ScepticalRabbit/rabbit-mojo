"""Compile one ordinary CPython extension with the supported Mojo module builder."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

class MojoBuild(build_ext):
    def build_extension(self, ext):
        compiler = os.environ.get("MOJO") or shutil.which("mojo")
        if not compiler:
            candidate = Path(sys.executable).parent / "mojo"
            if candidate.exists():
                compiler = str(candidate)
        if not compiler:
            raise RuntimeError("Mojo compiler missing: run uv pip install 'mojo==1.2.0', or set MOJO")
        target = Path(self.get_ext_fullpath(ext.name))
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([compiler, "build", "--emit", "shared-lib", "-O3",
                        "--target-cpu", "x86-64", ext.sources[0], "-o", str(target)], check=True)

setup(ext_modules=[Extension("rabbit_mojo._native", ["src/rabbit_mojo/_mojo/bindings.mojo"])],
      cmdclass={"build_ext": MojoBuild})
