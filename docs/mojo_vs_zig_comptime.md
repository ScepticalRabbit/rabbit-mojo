# Mojo versus Zig: the final four pairs

Verified with Mojo **1.2.0 (2016d0cd)** and Zig **0.16.0**. All four `zig test` commands passed, and `tools/test_zig.py` parses their printed numerical values and compares them against calls to the compiled Mojo extension. Zig is optional and never participates in the Python package build or runtime.

```bash
zig test examples/zig/ex25_COMPTIME_CONSTANTS.zig
zig test examples/zig/ex26_COMPTIME_GENERIC_VECTOR.zig
zig test examples/zig/ex27_COMPTIME_MATRIX_KERNEL.zig
zig test examples/zig/ex28_COMPTIME_UNROLLING.zig
.venv/bin/python tools/test_zig.py
```

The command wrapper reports an explicit untested status when Zig is absent. It does not skip any Python example. Each Zig file has a complete independent test with the same input values as its Python partner. Floating results are compared at absolute 1e-13 except generic float32, which permits relative 1e-6. These examples' integer-valued results are exactly representable in both precisions.

## 25 — Constants, compile-time expressions and branches

Sources: [Mojo polynomial](../src/rabbit_mojo/_mojo/comptime.mojo), [Python](../examples/ex25_COMPTIME_CONSTANTS.py), [Zig](../examples/zig/ex25_COMPTIME_CONSTANTS.zig).

The following source fragments are shown side by side; each column belongs to its linked implementation.

| Mojo | Zig |
|---|---|
| `def polynomial[degree: Int, unroll: Bool](x: Float64) -> Float64:` | `fn polynomial(comptime degree: usize, x: f64) f64 {` |
| `comptime coefficients = (1.0,2.0,3.0,4.0)` | `const coefficients = [_]f64{ 1, 2, 3, 4 };` |
| `comptime if unroll:` | `if (comptime degree == 2 or degree == 3) {` |
| `comptime for k in range(degree+1):` | `inline for (0..degree + 1) \|k\| {` |
| `value = value*x+Float64(coefficients[degree-k])` | `value = value * x + coefficients[degree - k];` |

Degree, coefficient tuple/array and `degree+1`/`degree-k` expressions are known at compilation. Static tuple indexing in Mojo requires the compile-time loop index. The native branch selects loop form; Zig additionally checks supported degree at compilation and uses `@compileError` otherwise. Evaluation of these constant expressions and branch conditions is compile-time work. Horner multiplication by **x**, accumulating the result, is runtime work in both function interfaces. This demonstrates compile-time expression evaluation, not a claim that arbitrary Mojo programs can execute in its constant evaluator.

Mojo produces `polynomial[2,True]` and `[3,True]` for the unrolled path, plus `[2,False]` and `[3,False]` for the reference. Runtime Python `degree` chooses among existing instantiations; it does not compile a new kernel. Zig's degree parameter is `comptime`, while x is an ordinary runtime argument. Test inputs happen to be constants, so optimisation may additionally constant-fold them; that is distinct from requiring x at compile time.

| Runtime input | Degree | Mojo | Zig |
|---|---:|---:|---:|
| x=2 | 2 | 17 | 17 |
| x=2 | 3 | 49 | 49 |

Coefficients are `[1,2,3,4]` in ascending powers: `1 + 2x + 3x² (+ 4x³)`. Both tests assert these values. Fixed coefficient/degree variants trade flexibility for code size. Adding many degrees increases compilation and binary costs. Compile-time constants alone do not demonstrate a speed improvement.

## 26 — Type and fixed-vector parameterisation

Sources: [Mojo Vec](../src/rabbit_mojo/_mojo/comptime.mojo), [Python](../examples/ex26_COMPTIME_GENERIC_VECTOR.py), [Zig](../examples/zig/ex26_COMPTIME_GENERIC_VECTOR.zig).

| Mojo | Zig |
|---|---|
| `struct Vec[dtype: DType, N: Int](Movable):` | `fn Vec(comptime T: type, comptime N: usize) type {` |
| `var data: Array[Scalar[Self.dtype],Self.N]` | `return struct { data: [N]T, ... };` |
| `def dot(self, other: Self) -> Scalar[Self.dtype]:` | `fn dot(a: Self, b: Self) T {` |
| `comptime for i in range(Self.N):` | `inline for (0..N) \|i\|` |
| `value += self.data[i]*other.data[i]` | `value += a.data[i] * b.data[i];` |

Mojo struct parameters use square brackets and `Self`-qualified names. Zig returns a type from a comptime-parameterised function; `@This()` gives its Self alias. Element dtype/type and N are compilation inputs. Array contents and the sum accumulator remain runtime values. Mojo borrows the NumPy inputs then **copies the few elements** into two native owned fixed arrays. This is intentionally different from the NumPy zero-copy examples.

Six Vec/dot specialisations are exercised: `{float32,float64} × {2,3,4}`, corresponding to `{f32,f64}` in Zig. Python's dtype/length are inspected at runtime only to select the native specialisation. These educational dot functions require no SIMD promise: the compiler may optimise an unrolled scalar loop, but the explicit SIMD example is ex22.

| Type in both languages | N | a | b | Mojo and Zig dot |
|---|---:|---|---|---:|
| float32/f32, float64/f64 | 2 | [1,2] | [2,1] | 4 |
| float32/f32, float64/f64 | 3 | [1,2,3] | [3,2,1] | 10 |
| float32/f32, float64/f64 | 4 | [1,2,3,4] | [4,3,2,1] | 20 |

Both native implementations assert agreement, and the cross-language runner compares all six actual results. Float32 has lower precision for other inputs; exact agreement on these values does not imply equal precision in general. Supporting unbounded lengths/dtypes would multiply specialisations and compilation costs; dynamic arrays are more suitable there.

## 27 — Statically shaped matrix kernels

Sources: [Mojo multiply](../src/rabbit_mojo/_mojo/linalg.mojo), [Python](../examples/ex27_COMPTIME_MATRIX_KERNEL.py), [Zig](../examples/zig/ex27_COMPTIME_MATRIX_KERNEL.zig).

| Mojo | Zig |
|---|---|
| `def multiply[N: Int](a: View[.float64], b: View[.float64], dst: View[.float64]):` | `fn multiply(comptime N: usize, a: [N*N]f64, b: [N*N]f64) [N*N]f64 {` |
| `comptime for i in range(N):` | `inline for (0..N) \|i\| {` |
| `comptime for j in range(N):` | `inline for (0..N) \|j\| {` |
| `comptime for k in range(N):` | `inline for (0..N) \|k\|` |
| `value += a.get(i*N+k)*b.get(k*N+j)` | `dst[i*N+j] += a[i*N+k] * b[k*N+j];` |
| `dst.put(i*N+j,value)` | `return dst;` |

N, loop extents and logical index arithmetic are known at compilation; there are separate N=2 and N=3 implementations. Native Mojo input/output views still contain runtime physical strides, so static logical shape does not remove every indexing cost. Zig's values have fixed inline storage `[N*N]f64`; the standalone function returns a value. Both store matrices in logical row-major order and implement the same triple product. Entries are runtime inputs even though tests choose deterministic values.

| N | Runtime a | Runtime b | Mojo and Zig product |
|---:|---|---|---|
| 2 | [[1,2],[3,4]] | 2I | [[2,4],[6,8]] |
| 3 | [[1,2,3],[4,5,6],[7,8,9]] | 2I | [[2,4,6],[8,10,12],[14,16,18]] |

Independent tests and the cross-language runner assert every output entry. Python matmul dispatch checks shape then calls one of the two compiled kernels. Zig type signatures enforce storage length statically. Larger N creates O(N³) unrolled operations, so this pattern can substantially increase code size/compilation time; it is deliberately restricted to small matrices. No result establishes a general advantage over BLAS.

## 28 — Unrolling versus runtime looping

Sources: [Mojo polynomial](../src/rabbit_mojo/_mojo/comptime.mojo), [Python](../examples/ex28_COMPTIME_UNROLLING.py), [Zig](../examples/zig/ex28_COMPTIME_UNROLLING.zig).

| Mojo | Zig |
|---|---|
| `def polynomial[degree: Int, unroll: Bool](x: Float64) -> Float64:` | `fn polynomial(comptime degree: usize, comptime unroll: bool, x: f64) f64 {` |
| `comptime if unroll:` | `if (unroll) {` (unroll is comptime) |
| `comptime for k in range(degree+1):` | `inline for (0..degree + 1) \|k\|` |
| `value = value*x+Float64(coefficients[degree-k])` | `value = value * x + coefficients[degree - k];` |
| `else: ... for k in range(degree+1):` | `else { for (0..degree + 1) \|k\| ... }` |

Both versions use Horner evaluation, not a different polynomial algorithm. Degree=3 and the unroll flag select two separate functions; the flag is not a branch inside the generated arithmetic. The loop index/tuple element selection is static in the unrolled version. Mojo's runtime-loop reference copies the four constants into a native Array, allowing runtime indexing. x and the accumulator remain runtime values. The compiler can independently choose to unroll/constant-fold an ordinary loop at O3, so source-level unrolling does not prove different machine code or performance.

| Runtime x | Mojo unrolled | Mojo runtime loop | Zig unrolled | Zig runtime loop |
|---:|---:|---:|---:|---:|
| -1 | -2 | -2 | -2 | -2 |
| 0.5 | 3.25 | 3.25 | 3.25 | 3.25 |
| 2 | 49 | 49 | 49 | 49 |

The Python, Zig and cross-language checks assert equality within 1e-13. Static dispatch creates bounded variants and can reduce runtime control work, at the expense of compilation/code size. Runtime dispatch at the Python boundary is still present. A speed claim would require dedicated timings and machine-code inspection; these examples make none.

## ABI and tooling differences

Mojo supplies the supported CPython module builder and the native NumPy borrowing boundary used by this project. Zig examples are standalone learning comparators with independent `zig test` entry points. They are never linked into the Mojo extension and are not Python runtime fallbacks. Mojo uses origins and reference conventions for scoped views; these Zig algorithms use owned fixed-array values. Those storage/ABI choices affect costs independently of compile-time syntax.

Official references: [Mojo parameterisation](https://mojolang.org/docs/manual/parameters/), [Mojo Array](https://mojolang.org/docs/std/collections/array/Array/), [Zig 0.16 language reference](https://ziglang.org/documentation/0.16.0/), [Zig comptime](https://ziglang.org/documentation/0.16.0/#comptime), [Zig inline for](https://ziglang.org/documentation/0.16.0/#inline-for). Syntax was verified by the installed compilers, rather than copied from older language versions.
