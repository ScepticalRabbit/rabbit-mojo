# Public API reference

`import rabbit_mojo as rm`; all kernel arithmetic runs in the extension. Public wrappers validate ndarray dtype, alignment, shape and backing-storage bounds. Output arrays are NumPy-owned; views borrow storage synchronously. Tiny configuration and collection conversions copy values explicitly.

| Function | Inputs and result |
|---|---|
| `scalar_roundtrip(integer, real, flag, text, optional=None)` | Signed Int32-range Python int → native 64-bit Int and Int32; float → f64/f32; bool inverted; native UTF-8 string suffix; Optional value doubled or None. Returns heterogeneous Python tuple. Overflow is rejected before conversion. |
| `collection_roundtrip(items, coordinate, settings, mode=Mode.ADD)` | Numeric Python list, numeric coordinate pair, `{'scale': number}`, Mode enum. Returns `(scaled_native_sum, coordinate_sum, enum_integer, retained_dict, converted_values)`. Last item is `(copied_native_list, copied_native_dict, heterogeneous_native_record)`. Fourth item is the original Python wrapper, demonstrating identity separately from conversion. Mode is a representation demonstration, not a selector for this reduction. |
| `Counter(initial)` | Bound Mojo-owned state; `.increment()`, `.value()`, static `.zero()`. Signed native Int overflow is rejected. Constructor positional only. |
| `native_affine(value, **kwargs)` | Native scalar affine calculation with optional `factor=2`, `offset=0`; other keywords fail in Mojo. |
| `native_error(message)` | Always raises native Error → Python Exception, for binding demonstration. |
| `scale_inplace(a, factor=2)` | Rank 1–4, f32/f64/int64; modifies original allocation. Official NumPyView for C-contiguous ranks 1–3; explicit signed-stride view otherwise. Integer factor/product overflow rejected. |
| `dot(a,b,*,simd=False)`, `norm(a)` | f64 1D reduction, including empty input (zero). Explicit SIMD uses width 4 and requires contiguous a/b; scalar accepts strides. `norm` composes the native dot with a Python scalar square root. |
| `add(a,b)`, `cross(a,b)`, `scale_vector(a,factor)`, `normalize(a)` | Native fixed-size f64 values; lengths 2/3/4 except cross requires 3; normalisation rejects zero norm. Returns new ndarray. |
| `matmul(a,b)`, `matvec(a,b)`, `transpose(a)`, `determinant(a)`, `inverse(a)` | f64 2×2/3×3; product dimensions compiled separately. Inverse/determinant use partial pivoting and relative 1e-14 pivot threshold; near singular inverse fails, determinant returns 0 at this threshold. |
| `transform_points(points,transform,out=None,*,vectors=False,simd=False)` | f64 `(N,3)` and affine `(4,4)` or Transform3D. New output unless supplied; exact in-place allowed; partial overlap rejected. |
| `transform_vectors(vectors,transform,out=None,*,simd=False)` | Same batch layout, no translation. |
| `rotate_vectors(vectors,rotation,out=None)` | `(N,3)` and `(3,3)` linear map; caller is responsible for providing a rotation when rotation semantics are needed. |
| `rotate_tensors(tensors,rotation,out=None)` | Shape ends `(3,3)`; computes `R A Rᵀ`; R must be proper orthogonal. |
| `parallel_transform(points,transform,*,workers=2)` | Native POSIX threads; 1..64 requested workers, capped to nonempty row count. Output is newly allocated contiguous storage. Inputs can be strided. Empty input returns empty output. |
| `tensor_invariants(a)` | Finite symmetric 3×3, optionally batched. Dictionary `I1`, `I2`, `I3`, `volumetric`, `deviator`, `norm`, `J2`. `volumetric` is the scalar spherical coefficient `tr(A)/3`, not a full matrix. |
| `principal_strains(a,*,directions=False)` | Finite symmetric 2×2/3×3, optionally batched; descending values, optionally `(values, column_axes)`. Native Jacobi, up to 100 iterations. |
| `maximum_engineering_shear(a)` | Native eigenvalues followed by Python scalar/array difference of largest and smallest values. |
| `material_roundtrip(material)` | Explicit dataclass → native Material → new Python dataclass, preserving E and nu. |
| `lame_parameters(material)` | IsotropicElastic(E,nu) → native `(lambda,mu)`; finite E>0 and -1<nu<0.5. Revalidates mutable dataclass each call. |
| `elastic_stress(strain,material)` | Tensorial symmetric strain → `(stress,energy_density)` from native Hooke law. Scalar energy for one tensor, ndarray for a batch. |
| `elastic_stiffness(material)` | Native 6×6 Voigt matrix for `[xx,yy,zz,yz,xz,xy]` and engineering shear input `[εxx,εyy,εzz,2εyz,2εxz,2εxy]`. |
| `differentiate(y,spacing,*,axis=-1,method='central')` | Uniform grid; forward/backward first-order interiors, central second-order interiors, or `second` second derivative. Central uses second-order one-sided boundary formulas. Forward/backward use their first-order formula wherever the required neighbour exists, with the opposite second-order one-sided formula at the final/initial edge. Second derivative uses four-point one-sided boundaries. At least 3 or 4 samples, respectively. |
| `gradient2d(y,spacing=(1,1))` | Two native first derivative calls; results follow axis order `(d/daxis0, d/daxis1)`. |
| `integrate(y,x=None,*,spacing=1,axis=-1,method='trapezoid')` | Integrates selected axis; increasing finite x, or positive uniform spacing. Simpson additionally needs odd sample count and uniform x. Returns scalar or array with selected axis removed. |
| `newton(initial,options=None,*,function='square_minus_two',raise_on_failure=False)` | Native selected functions: x²−2, cos(x)−x, x³−1 (`cube_minus_one`). NewtonOptions(atol=1e-12, rtol=1e-12, max_iterations=50). NewtonResult(root,converged,iterations,residual). Absolute residual must meet atol; step tolerance also uses rtol. Zero derivative raises native Error; exhausted iterations return converged=False or raise RuntimeError if requested. |
| `least_squares(a,b)` | Finite design `(m,n)`, m≥n≥1; response `(m,)`. Native Householder QR copies to native work buffers and back-substitutes. Rejects pivot-column norms ≤ `1e-12 * max(abs(a)) * m`. No column pivoting; use NumPy/SVD for difficult conditioning. |
| `interpolate(field,coordinates,*,origin=None,spacing=None,policy='raise',method='linear')` | Rank 1/2/3 regular scalar grid, at least 2 samples per axis; queries `(N,rank)` in array-axis order. Native 1D linear/2D bilinear/3D trilinear or nearest; origin default 0, spacing default 1. Returns `(N,)`. |
| `sample_image(image,xy,*,method='bilinear',policy='raise')` | Grey f64 image `[y,x]`; query `(N,2)` with `(x,y)` at integer pixel centres. Bilinear or nearest (ties floor(x+0.5)); policies raise/clamp. Strided image and query arrays are borrowed. |
| `q4_shape(xi,eta)` | Returns `(N, derivatives)` shaped `(4,)`, `(4,2)`. Reference node order `(-1,-1),(1,-1),(1,1),(-1,1)`. |
| `q4_integrate(vertices,*,powers=(0,0))` | Physical `(4,2)` counter-clockwise vertices; integrates x^px y^py with 2×2 Gauss. Returns integral and coordinates/determinants/Jacobians at Gauss points. Positive Jacobian required at corners and integration points. Polynomial exactness depends on geometry and degree; tests use rectangular geometry. |
| `polynomial(x,*,degree=2,unroll=True)` | Coefficients [1,2,3,4], compiled degree 2/3. Runtime x; explicit unrolled or runtime-loop variant. |
| `generic_dot(a,b)` | float32/64 lengths 2/3/4; six fixed-size Vec instantiations. Borrow inputs, copy their few elements into native owned fixed arrays. |

The native extension is private. Calling `_native` with unchecked pointers/shapes bypasses the memory-safety boundary. It is used only by documented diagnostic tests/benchmarks with already-validated arrays.

Floating point arithmetic is educational and finite-range: these algorithms do not promise overflow-proof norms, extreme-condition eigenproblems or a general robust nonlinear solver. Tests use scale-aware tolerances; eigenvalues are tested at scales 1e-15, 1 and 1e15, including repeated/nearly repeated roots. Benchmark and NumPy comparisons use double precision except the deliberate f32 examples.
