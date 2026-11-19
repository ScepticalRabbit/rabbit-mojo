"""Install a wheel into a temporary target and run without Mojo or the source tree.

Uses the existing uv-managed Python, not a second virtual environment. The base
interpreter runs with -I; only stdlib and the freshly installed target are visible.
"""

from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    wheels = sorted((root / "dist").glob("rabbit_mojo-2026.11.0-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Build exactly one 2026.11.0 wheel in dist first")
    with tempfile.TemporaryDirectory(prefix="rabbit-wheel-") as directory:
        target = Path(directory) / "site"
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--target",
                str(target),
                "--only-binary=:all:",
                str(wheels[0]),
            ],
            check=True,
        )
        code = """
from pathlib import Path
import importlib.util
import os
import runpy
import subprocess
import sys
sys.path.insert(0,sys.argv[1])
assert importlib.util.find_spec('mojo') is None, 'Compiler leaked into clean interpreter'
assert importlib.util.find_spec('modular') is None, 'Developer runtime leaked in'
import numpy as np
import rabbit_mojo as rm
assert Path(rm.__file__).is_relative_to(sys.argv[1])
assert rm.__version__=='2026.11.0'
a=np.arange(9.).reshape(3,3)
np.testing.assert_array_equal(rm.transform_points(a,np.eye(4)),a)
np.testing.assert_array_equal(rm.parallel_transform(a,np.eye(4)),a)
np.testing.assert_allclose(rm.principal_strains(np.eye(3)),[1,1,1])
extension=rm._native.__file__
linkage=subprocess.check_output(['ldd',extension],text=True)
assert 'not found' not in linkage
for name in ('libKGENCompilerRTShared.so','libMSupportGlobals.so','libAsyncRTRuntimeGlobals.so'):
    line=next(line for line in linkage.splitlines() if name in line)
    assert sys.argv[1] in line,line
print('Clean wheel:',rm.__version__,rm.__file__)
print(linkage)
for script in sorted(Path(sys.argv[2]).glob('ex[0-9][0-9]_*.py')):
    runpy.run_path(str(script),run_name='__main__')
print('PASS: clean wheel import, native execution, all 28 examples, bundled linkage, no Mojo compiler')
"""
        # Explicit environment snapshot also avoids native getenv changes leaking.
        import os

        environment = dict(os.environ)
        for key in (
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTHONEXECUTABLE",
            "MOJO_PYTHON_LIBRARY",
            "LD_LIBRARY_PATH",
        ):
            environment.pop(key, None)
        environment["PATH"] = "/usr/bin:/bin"
        base = getattr(sys, "_base_executable", sys.executable)
        result = subprocess.run(
            [base, "-I", "-c", code, str(target), str(root / "examples")],
            cwd=directory,
            env=environment,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            sys.stderr.write(result.stdout + result.stderr)
            raise RuntimeError("clean wheel verification failed")
        lines = result.stdout.splitlines()
        print(
            "\n".join(
                line
                for line in lines
                if line.startswith(("Clean wheel:", "PASS:", "\t"))
            )
        )


if __name__ == "__main__":
    main()
