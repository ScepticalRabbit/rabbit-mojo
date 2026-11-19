# Rabbit-Mojo: Scientific Python–Mojo Interoperability and Numerical Examples

**Status:** Implementation specification for coding agent  
**Project name:** `rabbit-mojo` (distribution), `rabbit_mojo` (Python import)  
**Primary target:** Linux x86-64; current stable CPython, NumPy, Mojo toolchain available locally  
**Scope:** All P0 and P1 examples; optional Zig paired examples at the end  
**Date:** 19 November 2026

## 1. Purpose and success criteria

Build a small, well-tested, educational Python package in which computational kernels are written in compiled **Mojo** and exposed through ordinary **Python** imports. Demonstrate the key Mojo language concepts and the practical interoperation patterns needed in **PyVale** (tensor and field calculations, scientific arrays, model validation) and **Riley** (image interpolation, geometry, SIMD). The package is a learning/reference project, **not** a comprehensive mathematical library.

A user should be able to:

```python
import numpy as np
import rabbit_mojo as rm

points = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float64)
transform = np.eye(4, dtype=np.float64)
transformed = rm.transform_points(points, transform)
```

This illustrates the **desired interface**; the agent must implement/document the actual available API. Avoid any Python fallback that silently replaces a purported Mojo example. Kernels must perform substantive computation as native Mojo operations, not through dynamically dispatched NumPy inside Mojo.

**The five non-negotiable deliverables:**

1. A working native Mojo extension importable from normal Python and installable via Python packaging tools.
2. Explicit Python-to-Mojo and Mojo-to-Python demonstrations for basic types, dataclasses, native structs and NumPy arrays.
3. Fully runnable, sequentially numbered `examples/exXX_CONCEPT.py` for **every P0 and P1 concept** listed below, plus the corresponding numerical tests.
4. A README with exact instructions for environment, build, install, running one example and running **all** examples. The agent must execute these commands and fix them until they work on the supported local machine.
5. An advanced set of **paired Mojo and Zig** compile-time examples at the **end of the example sequence**, with side-by-side explanations and independently runnable Zig tests if Zig is available.

**Important feasibility rule:** Mojo's APIs, wheel support and Python bindings may change. Verify current APIs directly against installed compiler documentation and upstream official documentation. If a feature is not supported, create a minimal reproducible blocker, provide a safe narrower variant, and document the limitation. Never claim zero-copy, wheel portability, keyword argument support, dataclass auto-conversion or native error propagation without verifying them.

## 2. Architecture and boundaries

- `rabbit_mojo` is the public Python API; numerical kernels are implemented in Mojo.
- Python wrappers own input validation, NumPy dtype/shape checks and user-friendly exceptions where appropriate.
- The Mojo boundary is isolated and documented; small shared helpers are acceptable, but avoid building an independent binding framework.
- Use public Python dataclasses for simple configuration; map values explicitly to native Mojo structs. Demonstrate the alternative of binding a Mojo-native struct as a Python-visible class if supported.
- Borrow NumPy memory where safe and possible; use caller-provided output arrays to avoid allocations when useful.
- Clearly distinguish a **borrowed buffer**, a **copy**, a **Python object wrapper**, an **owned native allocation**, and a **view with shape/strides**.
- Prefer native Mojo scalar arithmetic, `SIMD` and parallel facilities where available. Include scalar reference variants for validation.
- Default to float64 for scientific examples; add float32 where it teaches specialisation, numerical precision or array interop.
- Do not mix Python object arithmetic into a purported optimized loop; isolate conversion to the boundary.
- No GPU examples required. Do not introduce MAX, CUDA, or other heavy runtimes merely to satisfy examples.
- Keep any Zig examples optional and independent of Python package compilation. They are learning comparators, **not** alternative runtime implementations of the Python public API.

### Suggested repository structure

```text
rabbit-mojo/
├── .gitignore
├── pyproject.toml
├── setup.py                      # only if needed for the Mojo extension build
├── README.md
├── uv.lock                       # if generated and appropriate
├── src/
│   └── rabbit_mojo/
│       ├── __init__.py
│       ├── types.py               # Python dataclasses/enums
│       ├── bindings.py            # wrappers and validation
│       ├── scalar.py
│       ├── linalg.py
│       ├── tensor.py
│       ├── geometry.py
│       ├── numerical.py
│       ├── interpolation.py
│       └── _mojo/
│           ├── bindings.mojo
│           ├── scalar.mojo
│           ├── linalg.mojo
│           ├── tensor.mojo
│           ├── geometry.mojo
│           ├── numerical.mojo
│           ├── interpolation.mojo
│           └── comptime.mojo
├── examples/
│   ├── ex01_PYTHON_SCALARS.py
│   ├── ...
│   ├── ex25_COMPTIME_CONSTANTS.py
│   ├── ex26_COMPTIME_GENERIC_VECTOR.py
│   ├── ex27_COMPTIME_MATRIX_KERNEL.py
│   ├── ex28_COMPTIME_UNROLLING.py
│   └── zig/                       # paired Zig implementations, never required by pip
│       ├── ex25_COMPTIME_CONSTANTS.zig
│       ├── ex26_COMPTIME_GENERIC_VECTOR.zig
│       ├── ex27_COMPTIME_MATRIX_KERNEL.zig
│       └── ex28_COMPTIME_UNROLLING.zig
├── tests/
│   ├── test_bindings.py
│   ├── test_numpy_views.py
│   ├── test_linalg.py
│   ├── test_tensor.py
│   ├── test_geometry.py
│   ├── test_numerical.py
│   ├── test_interpolation.py
│   └── test_examples.py
├── benchmarks/
│   ├── bench_mesh_transform.py
│   ├── bench_tensor.py
│   └── bench_image_sampling.py
└── docs/
    ├── mojo_python_interop.md
    ├── numpy_memory.md
    ├── mojo_vs_zig_comptime.md
    ├── build_and_distribution.md
    └── benchmark_results.md
```

Keep module organisation simple. A single compiled extension with internal Mojo source modules is acceptable if it simplifies cross-platform packaging. Build artefacts should not be checked into Git unless a specific reason is documented.

## 3. Build tooling and environment (mandatory)

**Use native Python tooling to manage the Python package**, with the Mojo compiler only for building native Mojo sources. There is an existing **uv-managed `.venv/` in the repository root**; use that environment and do not create a parallel Conda/Pixi virtual environment.

- Provide `pyproject.toml` with package metadata, `src` layout, Python dependencies, test/benchmark extras and a supported Python build backend (prefer `setuptools` unless another standard Python backend offers a demonstrably simpler integration).
- Create `setup.py` **if required** for a custom `build_ext` invocation of `mojo build`; do not create unnecessary setup machinery.
- Prefer `uv sync` (when the project is set up for it), `uv pip install -e .` for editable install, and `uv run ...` or `.venv/bin/python ...` for execution. Verify the exact commands in the local environment. Do not require a second Python environment.
- Determine how Mojo is installed or invoked locally (`mojo --version`, toolchain paths, dynamic linker requirements), record the version, and give actionable failure messages if not available.
- Determine proper CPython extension naming/ABI suffix. Build an importable extension using the documented Mojo Python binding mechanism, not a library that can only be accessed by an undocumented manual `ctypes` hack.
- Integrate compilation with the build process so ordinary Python build/install commands run the Mojo compiler when building from source. Avoid absolute paths and environment-specific machine assumptions.
- Generate wheels with `python -m build --wheel --no-isolation` or a verified equivalent in `.venv`, and test a **clean install of the built wheel without Mojo installed** when feasible. Do not claim this has worked unless it has been tested.
- Inspect linkage (`ldd` on Linux or equivalent) and document **all non-system runtime shared-library dependencies**. Package permitted dependencies correctly or explicitly document an unmet requirement; a wheel that imports only with a local developer Mojo install is **not** a proven standalone wheel.
- Initial supported platform can be Linux x86-64 only. Document unsupported operating systems and interpreter versions rather than claiming untested support.
- Keep Zig completely optional; only advanced examples need the Zig toolchain.

**README must show verified copy/paste commands for:**

1. Checking `.venv`/`uv`, Python and Mojo versions.
2. Installing development dependencies.
3. Building the Mojo extension.
4. Editable installation, if supported; otherwise the correct alternative.
5. Import smoke test.
6. Running **one numbered example** directly.
7. Running **every P0/P1 example in numeric order**, via a documented script or pytest parametrization.
8. Running pytest.
9. Building a wheel and testing that wheel.
10. Running optional Zig comparisons if Zig is installed.

A top-level `examples/run_all.py` or `python -m pytest tests/test_examples.py` runner may be used, but it must verify each example exits successfully and must not depend on the working directory accidentally. Examples themselves must be directly runnable (`uv run python examples/exXX_CONCEPT.py`), provided the documented build/install steps were followed.

## 4. Example requirements and naming

**Mandatory pattern:** `examples/exXX_CONCEPT.py`, where `XX` is a zero-padded two-digit, monotonically increasing number and `CONCEPT` is a concise uppercase underscore-delimited label. Example: `ex06_NUMPY_STRIDES.py`. Scripts must be independent, runnable teaching materials, not trivial wrappers that only print a status message.

Every example must:

- Begin with a short docstring explaining the problem, Mojo feature, Python interoperability and expected outputs.
- Import and call a **real compiled Mojo implementation** unless its sole purpose is explicitly to demonstrate binding failure/unsupported behaviour.
- Use small deterministic numerical inputs and print meaningful inputs and results.
- Include simple `assert`/`np.testing` checks or invoke a precise verification helper.
- Explain the boundary, data conversions, copies/borrows, error cases and relevant memory layout.
- Point to the matching Mojo implementation and, where applicable, matching test.
- Exit with status 0 when successful; no interactive input, no network and no reliance on inaccessible external datasets.
- Keep computation modest so **all examples** finish quickly in CI.

When the example introduces a performance concept, compare at least a correct scalar/reference implementation, but put longer measurements in `benchmarks/`, not in the example scripts.

### P0 examples: required core functionality

| Script | Main concept | Mandatory demonstration |
|---|---|---|
| `ex01_PYTHON_SCALARS.py` | Fundamental scalars | Python `int`, `float`, `bool`, `str`, `None` to/from native Mojo `Int` / fixed-width ints / floats / bool / string / optional; precision and overflow cases. |
| `ex02_PYTHON_COLLECTIONS.py` | Containers | Python `tuple`, `list`, `dict` and enum to/from Mojo-native representation; native conversion versus keeping a Python object; invalid input. |
| `ex03_DATACLASS_STRUCTS.py` | Objects and ownership | Python slotted dataclass ↔ Mojo struct fieldwise conversion; expose a native struct as a Python-visible type if supported; construction, methods, reuse and lifetime. |
| `ex04_NUMPY_CONTIGUOUS.py` | Native NumPy views | Borrow contiguous 1D/2D/3D `float64` NumPy arrays using supported `Span`/`NumPyView` APIs; pass input and output; prove in-place modification. |
| `ex05_NUMPY_POINTERS.py` | Raw buffers and safety | Get typed pointer into NumPy storage; pointer offset, shape, dtype, address/lifetime; float32, float64 and integer arrays; read-only boundaries. |
| `ex06_NUMPY_STRIDES.py` | Non-contiguous arrays | Offset slices, unit/nonunit strides, negative stride, Fortran order and a non-contiguous 2D view; safe explicit stride handling or documented minimal blocker (no silent copy). |
| `ex07_VECTOR_MATH.py` | Small vectors | Dot, cross, norm, normalisation, vector addition/scaling; native fixed-size value types; SIMD-friendly logic. |
| `ex08_MATRIX_MATH.py` | Small matrices | Matrix-vector and 2×2/3×3 matrix products, transpose, determinant, inverse; static dimensions and singular cases. |
| `ex09_MESH_TRANSFORM.py` | Geometry and batching | Apply 4×4 homogeneous transform to `(N,3)` coordinates; output optional; in-place and strided input where safe; conventions and aliasing. |
| `ex10_TENSOR_INVARIANTS.py` | Tensor maths | Symmetric 3×3 tensor trace, invariants, volumetric and deviatoric component, norm, batched computation. |
| `ex11_PRINCIPAL_STRAINS.py` | Eigenvalues | Principal strains 2D and 3D, maximum engineering shear, sorted values and optional directions; comparison to `numpy.linalg.eigh`. |
| `ex12_TENSOR_ROTATION.py` | Tensor fields | Rotate batched vectors and tensors, `R A Rᵀ`, invariant preservation, shape/layout and batch memory access. |
| `ex13_NUMERICAL_DIFF.py` | Stencils | Forward/backward/central differences, second derivative and 2D gradient, boundary conditions, convergence order. |
| `ex14_NUMERICAL_INTEGRATION.py` | Reductions | Trapezoidal uniform/nonuniform integration, Simpson integration, integration along selected axis. |
| `ex15_NEWTON_RAPHSON.py` | Iterative solver | Scalar Newton–Raphson, native callable/selected function, convergence result struct, tolerances, derivative singularities, exceptions. |

### P1 examples: required applied engineering functionality

| Script | Main concept | Mandatory demonstration |
|---|---|---|
| `ex16_VECTOR_FIELD_TRANSFORM.py` | Batched field processing | Rotate/transform a vector field separately from coordinate translations, multiple spatial samples, scalar/SIMD versions. |
| `ex17_ELASTIC_MATERIAL.py` | Configuration objects | `IsotropicElastic` Python dataclass (E, nu) mapped to native struct; stiffness/lambda/mu; Hooke's law stress and strain energy density; 3D conventions. |
| `ex18_FIELD_INTERPOLATION.py` | Structured grids | 1D linear, 2D bilinear and 3D trilinear interpolation from regular field arrays; shape/stride and out-of-bounds policies. |
| `ex19_FE_QUADRATURE.py` | Element integration | Q4 reference quadrilateral shape functions, 2×2 Gauss quadrature, Jacobian/determinant, integration of test polynomials. |
| `ex20_LEAST_SQUARES.py` | Regression/reductions | Fit a small linear model by least squares with well-conditioned data; show numerical conditioning and rank-deficient failure; compare `numpy.linalg.lstsq`. |
| `ex21_IMAGE_SAMPLING.py` | Riley-like image access | Nearest-neighbour and bilinear grey image sampling at floating pixel coordinates; out-of-bounds policy, strides, memory safety. |
| `ex22_SIMD_SPECIALISATION.py` | SIMD explicitly | Scalar and native SIMD vector reductions or transform kernels, tails, alignment, compile-time width where available; correct validation. |
| `ex23_PARALLEL_BATCH.py` | Parallelism | Sequential and parallel mesh/tensor batch operations; correctness, determinism, thread-safety, and measurable boundaries. |
| `ex24_BINDINGS_ERRORS.py` | API robustness | Public function registration, constructor/method bindings, keyword arguments if supported, native errors→Python exceptions, bad input and clean failure. |

**Completion expectation:** Every P0 and P1 row has a corresponding actual example script, compiled Mojo code, and tests. An unsupported subfeature may be demonstrated by a safe and explicit negative test, but may not cause the whole example to be skipped without explanation.

### Final advanced examples: Mojo comptime, paired with Zig

Put these **last** in the sequence because they are more advanced. Each Python example must exercise a native Mojo implementation. Supply matching `.zig` file(s) with the **same algorithm and numerical values**, and a Markdown comparison in `docs/mojo_vs_zig_comptime.md`. Zig should use the available local stable compiler APIs (do not paste outdated Zig syntax).

| Python example + matching Zig source | Mojo and Zig compile-time topic | Same computation to demonstrate |
|---|---|---|
| `ex25_COMPTIME_CONSTANTS.py` + `zig/ex25_COMPTIME_CONSTANTS.zig` | Compile-time constants, compile-time branches, compile-time evaluation; distinguish compile time from runtime | Polynomial coefficients/order selected at compile time, evaluated on runtime inputs. |
| `ex26_COMPTIME_GENERIC_VECTOR.py` + `zig/ex26_COMPTIME_GENERIC_VECTOR.zig` | Generic numerical type and fixed-size vector length; Mojo parameterization versus Zig `comptime T: type`, `comptime N` | Fixed-size `Vec[T, N]`-style dot product for float32/float64 and dimensions 2/3/4. |
| `ex27_COMPTIME_MATRIX_KERNEL.py` + `zig/ex27_COMPTIME_MATRIX_KERNEL.zig` | Compile-time shape specialization / generated fixed-size operations | Multiply small statically shaped 2×2 and 3×3 matrices; separate generated implementations and runtime work. |
| `ex28_COMPTIME_UNROLLING.py` + `zig/ex28_COMPTIME_UNROLLING.zig` | Compile-time loops/loop unrolling, static dispatch and runtime branching trade-offs | Fixed-order finite-difference stencil or fixed-size polynomial evaluation, with runtime data and compile-time loop extent. |

**Required teaching comparison for each pair:**

1. Show relevant Mojo source and relevant Zig source side-by-side in docs; identify syntactic differences.
2. Explain exactly **which values/types are known at compile time**, which values remain runtime inputs, and which functions/instantiations are produced.
3. Explain the analogue of Mojo parameterisation in Zig (`comptime`, generic type arguments, `inline for` or suitable idiom), using current compiler syntax.
4. Show a result from **both implementations** and assert the outputs agree to stated numerical tolerance.
5. Explain limitations: e.g., over-specialisation, code size, compilation cost, ABI/Python exposure versus standalone Zig use.
6. Differentiate specialisation/unrolling from **measured** performance benefit. Do not promise SIMD or speedups solely because comptime was used.
7. If Zig isn't installed, leave its build/test as optional but ensure source is complete and clearly provide an untested status. If Zig is installed, run `zig test` (or other appropriate command) for all four paired files and record successful commands in README.

**Do not attempt to call Zig from the Mojo Python extension.** Comparison is conceptual and numerical; linking Mojo and Zig in one binary is outside scope.

## 5. Maths conventions and verification details

### Matrix and geometric conventions

- Coordinates have shape `(N, 3)` and are treated as **column vectors** mathematically.
- Homogeneous transform `T` maps `[x, y, z, 1]ᵀ` to `[x', y', z', w']ᵀ`.
- For affine transforms require bottom row `[0,0,0,1]`, or explicitly implement and test perspective divide with `w' != 0`; do not silently ignore `w`.
- Tensor rotation: `A' = R A Rᵀ`. Test against identity, 90° rotation, reversibility and invariant preservation.
- Document row-major/column-major storage as distinct from mathematical vector convention.

### Tensor and material conventions

- Strain tensors use **tensorial** shear components; specify when reporting engineering shear `γij = 2 εij`.
- Principal strains: symmetric tensors, sorted descending, values checked against `numpy.linalg.eigh`, repeated/nearly repeated roots and small perturbations.
- Deviatoric tensor: `dev(A) = A - (tr A / 3) I` for 3D; verify trace is (approximately) zero.
- Define invariants unambiguously: `I1 = tr A`, `I2 = 1/2[(tr A)^2 - tr(A²)]`, `I3 = det A`; if including `J2`, define it separately as `0.5 dev(A):dev(A)`.
- Isotropic linear elasticity: `σ = λ tr(ε) I + 2 μ ε`, where `μ = E/[2(1+ν)]` and `λ = Eν/[(1+ν)(1-2ν)]`. Validate admissible material parameters, particularly near `ν=0.5`.

### Numerical methods

- Finite differences: verify expected error order as grid size decreases on analytical functions.
- Numerical integration: verify exact integrals of low-degree polynomials and specify uniform-grid Simpson parity constraints.
- Newton solver: return structured information (root, converged flag or documented exception, iterations, residual); cover zero derivative and nonconvergence.
- Least squares: demonstrate least-squares residual and numerical conditioning. Avoid naive normal equations as an unqualified robust approach; document algorithm and limitations.
- Interpolation: specify coordinates, pixel-centre convention for image sampling, and behavior outside domain.

## 6. Python types and Mojo-native types

The following must be explicitly covered by `ex01`–`ex03` and `ex17`:

| Python type | Required topic |
|---|---|
| `int` | Native signed / fixed-width integer mapping; size and overflow |
| `float` | `float64`, float32 conversion and precision |
| `bool` | Native branching and round-trip |
| `str` | Text conversion and Unicode demonstration |
| `None` | Optional values, absent arguments |
| `tuple` | Fixed shape coordinates and heterogeneous tuples |
| `list` | Typed native collection conversion (and whether copies are made) |
| `dict` | Typed key/value extraction and validation |
| `enum.Enum` | Chosen enum mapping and invalid enum behaviour |
| `dataclasses.dataclass` | Explicit fieldwise mapping into native struct, not automatic ABI sharing |
| Native Mojo `struct` | Bound Python-facing constructor/method where supported |
| `np.ndarray` | Zero-copy borrowing, shape/strides, dtype, ownership and readonly flags |

A dataclass example should use:

```python
from dataclasses import dataclass

@dataclass(slots=True)
class IsotropicElastic:
    youngs_modulus: float
    poissons_ratio: float
```

Also use `NewtonOptions` and/or `Transform3D` and demonstrate a struct with a numerical array member or a scoped view; explain lifetime explicitly. Compare cost of repeated dataclass field conversion with constructing/reusing a native object only when both paths are actually implemented.

## 7. NumPy zero-copy and memory safety (P0 gate)

**Before implementing most kernels, pass these tests:**

1. Borrow C-contiguous 1D, 2D and 3D NumPy float64 arrays using documented Mojo APIs; verify shape, dtype and identity of shared storage where observably possible.
2. Perform an **in-place** Mojo operation and verify Python sees changes to the original allocation.
3. Write to a **caller-provided** output array. Guard against aliasing where it would invalidate results.
4. Support or explicitly report limitations for float32, at least one integer type, read-only arrays, and empty arrays.
5. Demonstrate nonzero-offset slices, non-unit and negative strides, Fortran order, and a non-contiguous 2D view. Safe stride-aware access is the goal; unsupported layouts must fail explicitly rather than silently copy.
6. Check lifetimes: a raw pointer must not outlive the Python allocation it references; do not store an unowned pointer in a long-lived native struct.
7. Show correct handling of aligned/misaligned addresses (or documented fail-fast restrictions) and bounds/shape validations before native access.
8. Measure or observe any allocation/copy at the boundary. A pointer being available does not by itself prove the public API is copy-free.

If APIs support only contiguous views, do **not** invent a bogus generic strided memoryview: implement an explicitly validated pointer+shape+byte-stride wrapper for synchronous calls if sound, otherwise demonstrate the blocker and use a safe contiguous-only API. Ensure negative strides and zero-sized arrays are not passed to unsafe assumptions.

## 8. Mojo binding examples

Demonstrate using the **current supported Mojo Python extension binding mechanism** (for example, `PythonModuleBuilder` if that is current in the installed toolchain):

- Native module initialisation and function exports.
- Simple scalar and NumPy parameters/returns.
- Native structs with constructors, methods and static methods when supported.
- Dataclass field extraction and conversion.
- Optional arguments and keyword arguments where supported.
- Native errors surfaced as ordinary Python errors with useful messages.
- Clear handling of type mismatch and unsupported memory layout.
- The build step producing a CPython-importable extension with correct naming.

A tiny Python wrapper is allowed for validation and ergonomic API; keep it honest about what executes in Python versus Mojo.

## 9. Testing and benchmarking

### Automated tests

- Use `pytest` and `numpy.testing`.
- Every numerical operation requires a direct analytical case, a NumPy comparison and edge cases.
- Include float32/float64 tests where supported with scale-aware tolerances.
- Include invalid-shape, invalid-dtype, readonly, singular matrix, nonconvergence and memory-layout cases.
- Include numerical invariants (trace of deviator, tensor rotation invariance, homogeneous identity, etc.).
- Add `tests/test_examples.py` that runs **all 28 Python example scripts** (or imports and calls their `main()` functions) and fails on a nonzero exit or missing file. Keep output concise.
- Never mark required examples as passed merely because they printed an explanation of an unsupported function; track unsupported subfeatures explicitly.

### Benchmarks

- Benchmark mesh transformations, batched tensor/principal-strain evaluation and image sampling.
- Compare against vectorised NumPy and optionally Cython where practical.
- Compare binding overhead for a small array versus large arrays, dtype conversion, and total end-to-end runtime.
- Separate scalar Mojo loops, native SIMD and parallel implementations where present.
- Use warmup, repeated independent measurements and report median plus variability, array size, hardware, compiler flags, library versions and number of threads.
- Include an explicit benchmark correctness check before timing; avoid invalid comparisons such as a fast compiled loop versus a slow Python element loop described as "Mojo vs NumPy".

## 10. Implementation plan and milestones

**Milestone 1 — Establish reproducible build:** Inspect existing `.venv`; create `pyproject.toml`, optional `setup.py`, a native Mojo hello-world extension, import smoke test and first README commands. Verify the actual Python build path.

**Milestone 2 — Python types and native structs:** Implement and test `ex01`–`ex03`, plus appropriate native Mojo bindings and tests. Document conversions and ownership.

**Milestone 3 — NumPy gate:** Implement and test `ex04`–`ex06`, including zero-copy contiguous array borrowing, writable outputs, lifetime and stride limitations. Do not paper over failed NumPy support.

**Milestone 4 — Core maths P0:** Implement and test `ex07`–`ex15` with analytical and NumPy references, native Mojo computation and simple documentation.

**Milestone 5 — Engineering and performance P1:** Implement and test `ex16`–`ex24`. Add benchmark scripts and record results.

**Milestone 6 — Advanced paired comptime:** Implement and test `ex25`–`ex28` in Mojo. Add matching `examples/zig/exXX_*.zig` and `docs/mojo_vs_zig_comptime.md` with detailed side-by-side comparisons. Run optional Zig tests if compiler is present.

**Milestone 7 — Distribution & final QA:** Build wheel, inspect dependencies, perform clean installation smoke test where feasible. Run exact README commands, all 28 examples in order, pytest and optional Zig tests. Fix instructions or code until they agree. Record outstanding portability limitations.

### Agent workflow/reporting

At each milestone, report:

- What was implemented and which scripts run.
- Exact version of Mojo, Python and Zig (if used).
- Relevant build/test commands and concise pass/fail summary.
- Any API gap, unsupported concept or safety restriction.
- Next milestone.

Avoid massive repository scans and noisy test logs. Target only project code and tests. Do not introduce extra infrastructure without evidence it is needed. Complete the implementation in order; optimize later once basic correctness and interoperability are proven.

## 11. Final acceptance checklist

- [ ] `pyproject.toml` and optional `setup.py` allow supported Python tooling to build Mojo extension.
- [ ] Uses project `.venv` and `uv`; no second environment required.
- [ ] README contains **verified exact commands** to build, install, run examples, test and build a wheel.
- [ ] Import works in a normal Python script.
- [ ] All **28** numbered `exXX_CONCEPT.py` files exist and are meaningful runnable examples, covering all P0/P1 requirements and advanced pairs.
- [ ] Every example calls substantive native Mojo code and has verification.
- [ ] Python primitive types, lists/tuples/dicts/enums, dataclasses and native structs are demonstrated.
- [ ] NumPy pointer/view tests include in-place writes, shape, stride, dtype, readonly and ownership.
- [ ] Mathematical conventions and numerical tolerances are documented and tested.
- [ ] SIMD, parallelism and compile-time specialization are concretely demonstrated, not merely described.
- [ ] Four paired Zig sources exist and demonstrate the same advanced algorithms with numerical checks.
- [ ] `docs/mojo_vs_zig_comptime.md` clearly explains Mojo versus Zig compile-time features.
- [ ] Python tests and entire example sequence pass on supported local environment.
- [ ] Built wheel is tested for import and runtime linkage; standalone installation claims match evidence.
- [ ] README explicitly reports any unsupported or untested Mojo, platform or Zig features.

## 12. Documentation references for agent

Check the most current official references and reconcile syntax against the local toolchain:

- Mojo documentation: https://docs.modular.com/mojo/
- Mojo Python interop: https://docs.modular.com/mojo/manual/python/
- Mojo standard library API: https://docs.modular.com/mojo/stdlib/
- Python packaging: https://packaging.python.org/
- uv: https://docs.astral.sh/uv/
- Zig language reference: https://ziglang.org/documentation/master/

**Final instruction:** Deliver a functioning demonstration package, not only design notes or placeholder files. Each numbered script must teach a real concept and run using the exact procedure documented in README. Any unsupported feature must be called out with a minimal reproducible example and safe documented behaviour. The user will use this repository as a working reference for potential Mojo development in PyVale and Riley.
