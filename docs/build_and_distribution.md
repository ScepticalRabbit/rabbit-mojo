# Build and distribution evidence

## Supported local configuration

Verified 19 November 2026 on Linux x86-64 (Ubuntu userspace, glibc 2.39), CPython 3.14.7, NumPy 2.5.5, Mojo 1.2.0 (2016d0cd), uv 0.12.6 and Zig 0.16.0. Package version is **2026.11.0**. Python metadata restricts support to CPython's 3.14 series; Windows, macOS, ARM and other Python ABIs are untested/unsupported. A CPython shared library must be present (the uv interpreter includes it).

`pyproject.toml` uses setuptools with a src layout, NumPy runtime dependency, dev and benchmark extras, and build dependencies Mojo 1.2.0/patchelf. `setup.py` implements a small build_ext, invokes `mojo build --emit shared-lib -O3 --target-cpu x86-64`, and writes the path returned by setuptools `get_ext_fullpath`, giving the correct CPython extension suffix. Package initialisation exports `PyInit__native` through PythonModuleBuilder. No import hook, JIT or ctypes bridge is required at runtime.

Both `uv sync --extra dev --extra benchmark` and `uv pip install -e '.[dev,benchmark]'` were executed successfully in the existing `.venv`. Build isolation works: the compiler and ELF patcher are declared build requirements. After editing Mojo files, run `.venv/bin/python setup.py build_ext --inplace` or reinstall the editable package. Python-source edits are visible immediately under editable installation; Mojo-source edits require compilation.

`MOJO` may select a compiler executable; the version check rejects unsupported compilers and its `mojo-compiler` Python package must match. No absolute machine paths occur in the build configuration. The compiler's generated absolute RUNPATH is replaced after compilation. A lockfile records resolved Python dependencies. Zig is not a package/build dependency.

## Runtime linkage

Initial `ldd` inspection found a developer-install absolute RUNPATH and these non-system dependencies:

| Runtime library | Bundled source |
|---|---|
| libKGENCompilerRTShared.so | mojo-compiler 1.2.0, modular/lib |
| libMSupportGlobals.so | mojo-compiler 1.2.0, modular/lib |
| libAsyncRTRuntimeGlobals.so | mojo-compiler 1.2.0, modular/lib |

The build copies these three libraries into `rabbit_mojo/.libs`. It sets the extension RUNPATH to `$ORIGIN/.libs` and each library's RUNPATH to `$ORIGIN` using patchelf. Both editable installs and wheels include them. System dependencies observed transitively are libc, libm, libstdc++, libgcc_s and the ELF loader. Runtime libraries carry versioned glibc/libstdc++ requirements; the inspected binaries require glibc symbols up to **GLIBC_2.35**. This is a Linux wheel, **not** an audited manylinux wheel. Successful testing on glibc 2.39 does not prove execution on other distributions.

CPython interop also dynamically locates the running interpreter's libpython using sysconfig; this interpreter dependency is distinct from ELF NEEDED dependencies. `_bootstrap.py` selects it before import and restores loader-related environment variables afterwards. See the interop document for the local loader issue and regression test.

The project is MIT. Bundled Modular binary portions retain their upstream terms; human-readable MAX and Apache/LLVM licence texts and a NOTICE are installed in `rabbit_mojo/licenses`. The upstream [MAX License](https://github.com/modular/modular/blob/main/Licenses/MAX_LICENSE) permits redistribution in applications with material additional functionality while preserving its terms (sections 1.2 and 2); the [upstream licence overview](https://github.com/modular/modular/blob/main/LICENSE) identifies mixed licensing. No compiler, debugger, MAX Python package or GPU runtime is bundled. Binary build products and .libs are ignored by Git; build results are local artefacts.

## Wheel verification

Executed:

```bash
.venv/bin/python -m build --wheel --no-isolation
.venv/bin/python tools/verify_wheel.py
```

The generated wheel is `dist/rabbit_mojo-2026.11.0-cp314-cp314-linux_x86_64.whl`. The verifier uses `uv pip install --target` in a temporary directory, then invokes the existing interpreter's **base executable** with `-I`. This creates no parallel project environment. It removes developer library/interpreter environment overrides and restricts PATH to system commands. The interpreter sees stdlib and the installed target only. Assertions prove `mojo` and developer `modular` are not importable and rabbit_mojo loads from the new target. Native transform/principal-strain/thread operations and all 28 example scripts run against that installed package. `ldd` verifies all three Modular libraries resolve from the target's .libs and no library is missing.

Result: **PASS**, including normal import, substantive native execution, all examples and compiler-free runtime linkage. This proves a self-contained wheel with respect to Mojo dependencies on this machine, not cross-machine portability or static linking. A separate machine/container test remains outside the verified scope.

Source packaging includes Mojo modules, examples, Zig pairs, tests, tools, benchmark sources and docs via MANIFEST.in. The archive was inspected to verify all 28 Python examples and native sources were present and no .so files were included. An isolated `uv build --wheel --out-dir <temporary-directory> dist/rabbit_mojo-2026.11.0.tar.gz` succeeded, rebuilding native code from the source archive. No compiled extension is committed to Git.

## Verified commands

The README provides the complete copy/paste sequence. The following were additionally used during validation:

```bash
.venv/bin/python -m pytest tests/test_numpy_views.py -q
.venv/bin/python examples/run_all.py
.venv/bin/python -m pytest -q
.venv/bin/python tools/test_zig.py
.venv/bin/python tools/run_benchmarks.py
ldd src/rabbit_mojo/_native*.so
readelf -d src/rabbit_mojo/_native*.so
```

The installed compiler may print an informational Crashpad warning. The bound Counter currently produces a CPython deprecation warning for missing __module__ metadata. Neither is hidden or treated as a numerical success condition; all required examples execute native computation and assertions.
