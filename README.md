# rabbit-mojo — 2026.11.0

A working reference for scientific Python ↔ compiled Mojo interoperability: 28 checked examples, native numerical kernels, NumPy borrowing, explicit SIMD, POSIX threads, and four independent Zig comparisons. Python validates inputs; Mojo performs the kernel arithmetic. There is no Python numerical fallback.

Verified on **Linux x86-64**, CPython **3.14.7**, NumPy **2.5.5**, Mojo **1.2.0 (2016d0cd)**, uv **0.12.6**, Zig **0.16.0**, glibc **2.39**. Other Python versions, operating systems and wheel platforms are unsupported/untested. The extension targets baseline `x86-64`, not the developer CPU. This is an educational package, not a general numerical library.

## Build and run

Use the existing repository `.venv`; no Conda/Pixi environment is needed. Run these commands from the repository root. Mojo is a development/build dependency, not a wheel runtime requirement.

```bash
uv --version
.venv/bin/python --version
uv pip install 'mojo==1.2.0'
.venv/bin/mojo --version

# Install locked development and benchmark dependencies; also builds the extension.
uv sync --extra dev --extra benchmark

# Explicit rebuild after changing any .mojo source.
.venv/bin/python setup.py build_ext --inplace

# Editable installation through ordinary Python packaging (build isolation supported).
uv pip install -e '.[dev,benchmark]'

# Import and version smoke test.
.venv/bin/python -c 'import rabbit_mojo as rm; print(rm.__version__); print(rm.polynomial(2.))'

# One example, or all 28 in numeric order.
.venv/bin/python examples/ex09_MESH_TRANSFORM.py
.venv/bin/python examples/run_all.py

# Numerical, memory-safety, binding and example tests.
.venv/bin/python -m pytest -q

# Wheel, then clean installation without Mojo on the interpreter import path.
.venv/bin/python -m build --wheel --no-isolation
.venv/bin/python tools/verify_wheel.py

# Optional Zig: four independent tests plus comparisons with native Mojo results.
zig version
.venv/bin/python tools/test_zig.py

# Correctness-checked measurements and recorded results.
.venv/bin/python tools/run_benchmarks.py
```

`uv run --no-sync python examples/ex09_MESH_TRANSFORM.py` is equivalent after installation. Examples and their test runner resolve paths independently of the current directory. `verify_wheel.py` installs into a temporary target and uses the same uv-managed **base interpreter** with `-I`, without creating another virtual environment. It asserts that neither `mojo` nor the developer `modular` package is importable, checks `ldd`, and runs all 28 examples against the installed wheel.

A missing compiler produces an actionable build error. The build pins Mojo 1.2.0 because its binding API changes. Set `MOJO=/path/to/mojo` to choose that compiler explicitly; its matching `mojo-compiler` Python distribution must be installed. `patchelf` is included in build dependencies. An informational Crashpad diagnostic may appear from the installed compiler; it does not prevent compilation.

## Python API

```python
import numpy as np
import rabbit_mojo as rm

points = np.array([[0., 1., 2.], [3., 4., 5.]], dtype=np.float64)
transform = np.eye(4, dtype=np.float64)
transform[:3, 3] = [10., 20., 30.]
result = rm.transform_points(points, transform)
np.testing.assert_allclose(result, points + [10., 20., 30.])
```

Numerical arrays are native-endian float64 unless explicitly noted. Inputs are never silently cast or made contiguous. General kernels accept validated signed-stride views. `scale_inplace` also supports float32 and int64; `generic_dot` supports float32/float64. `dot(..., simd=True)` requires contiguous 1D arrays. Small vector operations support lengths 2/3/4; small matrices support 2×2/3×3. Caller-provided output is available for geometry operations. See [API details](docs/api.md).

| Examples | Topic / primary API |
|---|---|
| 01–03 | `scalar_roundtrip`, `collection_roundtrip`, `Counter`, dataclass → native struct |
| 04–06 | `scale_inplace`, official NumPyView, typed pointers, shape/strides, ownership |
| 07–09 | `dot`, `cross`, `normalize`, `matmul`, `inverse`, `transform_points` |
| 10–12 | `tensor_invariants`, `principal_strains`, `rotate_vectors`, `rotate_tensors` |
| 13–15 | `differentiate`, `gradient2d`, `integrate`, `newton` |
| 16–17 | `transform_vectors`, `IsotropicElastic`, `elastic_stress`, `elastic_stiffness` |
| 18–21 | `interpolate`, `q4_shape`, `q4_integrate`, `least_squares`, `sample_image` |
| 22–24 | Explicit SIMD, native `parallel_transform`, `native_affine` kwargs and binding errors |
| 25–28 | Compile-time polynomial, `Vec[dtype,N]`, matrix shapes and unrolled Horner loops, paired with Zig |

Every `examples/exXX_CONCEPT.py` contains independent deterministic inputs, numerical assertions, meaningful printed results, boundary explanations, and links to its Mojo source and tests. See [compile-time comparisons](docs/mojo_vs_zig_comptime.md) for the four final pairs and their verified results.

## Conventions and limitations

Coordinates are `(N,3)` and mathematically column vectors. Only affine 4×4 transforms with bottom row exactly `[0,0,0,1]` are accepted. Vectors exclude translation. Tensor rotation is `R A Rᵀ`; strain shear is **tensorial**, maximum engineering shear is `εmax − εmin`. Eigenvalues descend and eigenvectors are columns; repeated eigenvectors have no unique orientation. A proper orthogonal rotation is required for tensor rotation.

Native `Error` becomes a generic Python `Exception` with its message. Python validation raises `TypeError`, `ValueError` or `OverflowError`. Raw bound `scale` is positional-only; `native_affine` explicitly demonstrates supported native `**kwargs`. `Counter(initial)` is a positional native constructor, with methods and a `zero()` static method. Dataclasses are converted fieldwise, never automatically ABI-shared. Mojo 1.2 emits a harmless CPython deprecation warning that its bound `Counter` lacks `__module__`; this upstream binding limitation is retained visibly.

Raw pointers are synchronous and kept with their owners. Unsafe/misaligned/out-of-allocation views and overlapping writes fail validation. Partial input/output overlap is rejected; exactly `out=points` is safe for sequential transforms. SIMD reduction has scalar tails. Native parallelism uses Linux POSIX threads and joins before returning; it is not the MAX parallel runtime. The GIL stays held in the calling thread; worker threads touch no Python objects. See [memory safety](docs/numpy_memory.md) and [interop details](docs/mojo_python_interop.md).

Stencils assume positive uniform spacing; boundaries are one-sided. Simpson requires uniform spacing and an odd number of points. Newton selects one of three compiled functions, not an arbitrary Python callback. Least squares uses unpivoted Householder QR with rank checks, for small well-conditioned designs; it is not a replacement for a pivoted QR/SVD solver. Grid coordinates follow array-axis order; image coordinates use `(x,y)` at integer pixel centres. Out-of-bounds sampling raises unless explicitly clamped.

## Distribution and evidence

The wheel is `rabbit_mojo-2026.11.0-cp314-cp314-linux_x86_64.whl`. It bundles the three required Mojo runtime shared libraries with relative ELF search paths and upstream notices. It does **not** install the Mojo compiler or MAX Python runtime. Its bundled Modular portions retain their upstream licences; project sources are MIT. See [build and linkage](docs/build_and_distribution.md). The wheel passed clean installation/import/native computation/all-example checks on this machine. This is not a claim of manylinux or cross-machine ABI portability.

All 28 examples, the Python test suite, and four Zig tests/comparisons were executed successfully. [Benchmark results](docs/benchmark_results.md) include hardware, versions, flags, sizes, warmup, medians and variability. Most general stride-aware scalar kernels are slower than vectorised NumPy here; native thread transforms have a specialised direct-stride loop. Measurements distinguish those implementations rather than claiming an isolated threading speedup.
