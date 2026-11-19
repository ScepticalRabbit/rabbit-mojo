# Python ↔ Mojo boundary

The single extension is built with Mojo 1.2.0's supported [`PythonModuleBuilder`](https://mojolang.org/docs/std/python/bindings/PythonModuleBuilder/), not a ctypes library. `_mojo/bindings.mojo` exports `PyInit__native` with C ABI and registers functions via `def_function`. CPython imports `_native` using its normal ABI extension suffix. Python-facing parameters/returns are PythonObject wrappers; conversions into Int/Int32/Float64/Float32/Bool/String/Optional and typed collections happen once before native loops. Functions do not dynamically dispatch NumPy arithmetic.

Python int is unbounded; native Int is signed 64-bit on this target. The primitive demonstration deliberately admits only the Int32 range so that both Int and Int32 mappings are safe, rejecting overflow explicitly. Counter demonstrates native Int range. Native text is UTF-8 and the examples preserve Unicode. None maps to a native Optional and back. Native tuple, List and Dict values are copied and converted back; the collection example also returns a retained Python dict by identity, clearly distinguishing wrapper from native conversion. The enum maps explicitly to an Int after Python enum validation.

IsotropicElastic is a slotted Python dataclass. Material is a separate native struct; E and nu are extracted fieldwise. Stress and stiffness arithmetic execute natively. Mutable configurations are revalidated on use. NewtonOptions maps to native arguments; native SolveResult converts to the frozen public NewtonResult dataclass. Transform3D has a numerical array member retained by Python; View retains that owner only for a synchronous call. There is no automatic dataclass ABI conversion.

Counter is a Python-visible native type registered with `add_type`, `def_py_init`, `def_method` and `def_staticmethod`. Python construction allocates an owned Mojo value and its state is reused across calls. `.zero()` demonstrates native ownership transfer with `PythonObject(alloc=...)`. Scalar state needs no borrowed pointer. No benchmark claims a conversion-vs-reuse advantage for Material: a reusable native Material API has not been implemented. Counter demonstrates actual reusable native state independently.

## Verified support and API gaps

| Feature | Verified behaviour |
|---|---|
| ordinary extension import | CPython 3.14, setuptools editable/wheel builds |
| native type constructor/method/static | Counter(initial), increment(), value(), zero() |
| native keyword arguments | `native_affine(3., factor=2., offset=1.) == 7.`; `var **kwargs: PythonObject` is represented as Mojo StringDict[PythonObject] |
| positional functions | `def_function` without **kwargs rejects named arguments; Python wrappers supply ergonomic keywords |
| optional values | Native Optional handles explicit None; public wrapper supplies omitted default |
| native failures | Mojo Error is a generic Python Exception with its message, not a specialised ValueError |
| dataclasses | Explicit field conversion only |
| arbitrary strides | Official NumPyView is contiguous-only; validated custom synchronous view supplies stride support |
| parallel runtime | Native POSIX threads via std.ffi; no MAX dependency |
| type metadata | Current builder omits Counter.__module__, triggering a CPython deprecation warning during import |

`examples/ex24_BINDINGS_ERRORS.py` and `tests/test_bindings.py` are runnable minimal reproductions of positional keyword rejection, native error type and bad constructor calls; they also demonstrate the safe supported keyword variant. Unsupported subfeatures do not skip their examples. The native constructor's null kwargs check follows the installed stdlib builder's `_obj_ptr` check; this small low-level dependency is isolated in scalar.mojo and tested against the pinned compiler.

## Interpreter loader workaround

Mojo 1.2's Python loader otherwise discovers `/usr/bin/python3` on this machine and modifies the C environment's PYTHONEXECUTABLE/PYTHONPATH. In a Python extension host that can cause subprocesses to lose the caller's venv, and may choose the wrong Python shared library. `_bootstrap.py` sets MOJO_PYTHON_LIBRARY to the running interpreter's sysconfig library and PYTHONEXECUTABLE to sys.executable before import, then restores the original environment with explicit unsetenv calls (Python's os.environ cache does not observe native setenv). A shared libpython is required; a missing library gives an explicit ImportError. This workaround uses no ctypes calls.

`tests/test_examples.py` runs examples in fresh subprocesses from unrelated working directories, covering this failure. Wheel verification uses the base interpreter in isolated mode and no developer module path. See [distribution details](build_and_distribution.md).

Official references checked against actual compilation: [calling Mojo from Python](https://mojolang.org/docs/manual/python/mojo-from-python/), [Python bindings](https://mojolang.org/docs/std/python/bindings/), [NumPy helpers](https://mojolang.org/docs/std/python/numpy/) and the [upstream stdlib binding source](https://github.com/modular/modular/blob/main/Mojo/stdlib/std/python/bindings.mojo).
